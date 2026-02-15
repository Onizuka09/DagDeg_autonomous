# zetta_protocol.py
import serial
import time
import struct
import threading
from queue import Queue
from typing import Optional, Callable, Any, Union
from dataclasses import dataclass
from enum import IntEnum
from crc import Calculator, Configuration

# CRC Configuration
_crc_config = Configuration(
    width=8,
    polynomial=0x07,
    init_value=0xFFFFFFFF,
    final_xor_value=0x00000000,
    reverse_input=False,
    reverse_output=False,
)

class ZettaPacketType(IntEnum):
    """Packet types for Zetta protocol"""
    MSG_ACK = 0
    MSG_PUBLISH = 1
    MSG_SUBSCRIBE = 2

@dataclass
class ZettaPacket:
    """Container for parsed Zetta packet"""
    type: ZettaPacketType
    data: bytes
    timestamp: float
    raw_packet: bytes  # Original packet bytes for debugging

class ZettaProtocol:
    """
    Zetta Protocol Python Implementation
    
    A modular packet protocol library for embedded communication.
    Users can define custom packet handlers for parsing and creating packets.
    """
    
    # Protocol constants
    START_BYTE = 0xAA
    STOP_BYTE = 0xBC
    MAX_PAYLOAD_SIZE = 40
    
    def __init__(self, 
                 port: str = "/dev/ttyACM0", 
                 baudrate: int = 115200, 
                 timeout: float = 0.1,
                 rx_callback: Optional[Callable[[ZettaPacket], None]] = None,
                 error_callback: Optional[Callable[[str], None]] = None):
        """
        Initialize Zetta Protocol instance.
        
        Args:
            port: Serial port (e.g., "/dev/ttyACM0", "COM3")
            baudrate: Baud rate for serial communication
            timeout: Serial read timeout in seconds
            rx_callback: Optional callback function for received packets
            error_callback: Optional callback function for errors
        """
        self.ser = serial.Serial(port, baudrate, timeout=timeout)
        self.calculator = Calculator(_crc_config)
        self.rx_queue = Queue(10)
        self.rx_callback = rx_callback
        self.error_callback = error_callback
        self.stop_threads = False
        self._rx_thread = None
        self._lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'packets_sent': 0,
            'packets_received': 0,
            'crc_errors': 0,
            'frame_errors': 0,
            'bytes_received': 0,
        }
        
        # User-defined packet handlers
        self._packet_handlers = {}
        
    def start(self):
        """Start the receiver thread"""
        if self._rx_thread is None or not self._rx_thread.is_alive():
            self.ser.reset_input_buffer() 
            self.stop_threads = False
            self._rx_thread = threading.Thread(target=self._receiver_thread, daemon=True)
            self._rx_thread.start()
            print(f"Zetta Protocol started on {self.ser.port}")
    
    def stop(self):
        """Stop the receiver thread and cleanup"""
        self.stop_threads = True
        if self._rx_thread:
            self._rx_thread.join(timeout=1.0)
        try:
            if self.ser.is_open:
                # Cancel any pending read/write operations
                self.ser.cancel_read() 
                self.ser.close()
        except:
            print("error closing the serial")
            pass
        print("Zetta Protocol stopped")
    
    def register_packet_handler(self, 
                               packet_type: ZettaPacketType, 
                               parser: Callable[[bytes], Any],
                               builder: Optional[Callable[[Any], bytes]] = None):
        """
        Register a custom packet handler for a specific packet type.
        
        Args:
            packet_type: Packet type to handle
            parser: Function to parse payload bytes into user data
            builder: Optional function to convert user data to payload bytes
        """
        self._packet_handlers[packet_type] = {
            'parser': parser,
            'builder': builder
        }
    
    def send_raw(self, packet_type: ZettaPacketType, payload: bytes) -> bool:
        """
        Send raw bytes as a Zetta packet.
        
        Args:
            packet_type: Type of packet to send
            payload: Raw payload bytes (max 25 bytes)
            
        Returns:
            True if packet was sent successfully
        """
        if len(payload) > self.MAX_PAYLOAD_SIZE:
            self._handle_error(f"Payload too large: {len(payload)} > {self.MAX_PAYLOAD_SIZE}")
            return False
        
        try:
            packet = self._create_packet(packet_type, payload)
            with self._lock:
                self.ser.write(packet)
                self.stats['packets_sent'] += 1
            return True
        except Exception as e:
            self._handle_error(f"Send failed: {e}")
            return False
    
    def send(self, packet_type: ZettaPacketType, data: Any) -> bool:
        """
        Send structured data using registered packet builder.
        
        Args:
            packet_type: Type of packet to send
            data: Data to send (will be converted using registered builder)
            
        Returns:
            True if packet was sent successfully
        """
        if packet_type not in self._packet_handlers:
            self._handle_error(f"No builder registered for packet type {packet_type}")
            return False
        
        handler = self._packet_handlers[packet_type]
        if not handler['builder']:
            self._handle_error(f"No builder function for packet type {packet_type}")
            return False
        
        try:
            payload = handler['builder'](data)
            return self.send_raw(packet_type, payload)
        except Exception as e:
            self._handle_error(f"Failed to build packet: {e}")
            return False
    
    def get_packet(self, timeout: Optional[float] = None) -> Optional[ZettaPacket]:
        """
        Get a received packet from the queue (blocking).
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            ZettaPacket if available, None if timeout
        """
        try:
            return self.rx_queue.get(timeout=timeout)
        except:
            return None
    
    def process_packet(self, packet: ZettaPacket) -> Any:
        """
        Process a received packet using registered parser.
        
        Args:
            packet: ZettaPacket to process
            
        Returns:
            Parsed data if parser exists, raw payload otherwise
        """
        if packet.type in self._packet_handlers:
            handler = self._packet_handlers[packet.type]
            if handler['parser']:
                try:
                    return handler['parser'](packet.data)
                except Exception as e:
                    self._handle_error(f"Failed to parse packet: {e}")
        
        # Return raw data if no parser registered
        return packet.data
    
    def flush(self):
        """Flush the receive queue"""
        while not self.rx_queue.empty():
            try:
                self.rx_queue.get_nowait()
                self.rx_queue.task_done()
            except:
                break
    
    def get_stats(self) -> dict:
        """Get communication statistics"""
        return self.stats.copy()
    
    # Internal methods
    def _create_packet(self, packet_type: ZettaPacketType, payload: bytes) -> bytes:
        """Create a Zetta protocol packet"""
        packet = bytearray()
        packet.append(self.START_BYTE)
        packet.append(packet_type.value)
        packet.append(len(payload))
        packet.extend(payload)
        
        # Calculate CRC (type + len + payload)
        crc_data = packet[1:]  # Everything after START byte
        checksum = self.calculator.checksum(crc_data) & 0xFF
        packet.append(checksum)
        packet.append(self.STOP_BYTE)
        
        return bytes(packet)
    
    def _parse_packet(self, raw_packet: bytes) -> Optional[ZettaPacket]:
        """Parse a raw packet and validate"""
        if len(raw_packet) < 6:
            return None
        
        if raw_packet[0] != self.START_BYTE or raw_packet[-1] != self.STOP_BYTE:
            return None
        
        pkt_type_value = raw_packet[1]
        length = raw_packet[2]
        
        if len(raw_packet) != 5 + length:
            self.stats['frame_errors'] += 1
            return None
        
        payload = raw_packet[3:3+length]
        crc_received = raw_packet[-2]
        
        # Calculate CRC (type + len + payload)
        crc_calculated = self.calculator.checksum(raw_packet[1:3+length]) & 0xFF
        
        if crc_received != crc_calculated:
            self.stats['crc_errors'] += 1
            return None
        
        try:
            packet_type = ZettaPacketType(pkt_type_value)
        except ValueError:
            packet_type = ZettaPacketType(pkt_type_value)  # Unknown type
        
        return ZettaPacket(
            type=packet_type,
            data=payload,
            timestamp=time.time(),
            raw_packet=raw_packet
        )
    
    def _receiver_thread(self):
        """Thread for continuous packet reception"""
        buffer = bytearray()
        
        while not self.stop_threads:
            try:
                # Read available bytes
                if self.ser.in_waiting > 0:
                    with self._lock:
                        data = self.ser.read(self.ser.in_waiting)
                    self.stats['bytes_received'] += len(data)
                    buffer.extend(data)
                
                # Process complete packets in buffer
                while len(buffer) >= 6:  # Minimum packet size
                    # Find START byte
                    if buffer[0] != self.START_BYTE:
                        buffer.pop(0)
                        continue
                    
                    # Check if we have enough data for minimal packet
                    if len(buffer) < 5:  # start + type + len + crc + stop
                        break
                    
                    length = buffer[2]
                    expected_size = 5 + length
                    
                    if len(buffer) >= expected_size:
                        # Extract complete packet
                        raw_packet = bytes(buffer[:expected_size])
                        buffer = buffer[expected_size:]
                        
                        # Parse and validate packet
                        packet = self._parse_packet(raw_packet)
                        if packet:
                            if self.rx_queue.full():
                                try:
                                    self.rx_queue.get_nowait()
                                except:
                                    pass
                            self.stats['packets_received'] += 1
                            # self.rx_queue.put(packet)
                            self.rx_queue.put(packet)
                            # Call user callback if registered
                            if self.rx_callback:
                                try:
                                    self.rx_callback(packet)
                                except Exception as e:
                                    self._handle_error(f"RX callback error: {e}")
                    else:
                        # Not enough data for complete packet
                        break
                
                time.sleep(0.001)  # Prevent CPU hogging
                
            except Exception as e:
                self._handle_error(f"Receiver thread error: {e}")
                time.sleep(0.1)
    
    def _handle_error(self, message: str):
        """Handle error messages"""
        print(f"[Zetta Error] {message}")
        if self.error_callback:
            try:
                self.error_callback(message)
            except:
                pass

# Helper functions for common packet types
def create_struct_parser(format_str: str):
    """Create a parser for structured data using struct format"""
    def parser(payload: bytes):
        return struct.unpack(format_str, payload)
    return parser

def create_struct_builder(format_str: str):
    """Create a builder for structured data using struct format"""
    def builder(data):
        return struct.pack(format_str, *data)
    return builder

def create_string_parser(encoding: str = 'utf-8'):
    """Create a parser for string data"""
    def parser(payload: bytes):
        return payload.decode(encoding, errors='ignore')
    return parser

def create_string_builder(encoding: str = 'utf-8'):
    """Create a builder for string data"""
    def builder(data: str):
        return data.encode(encoding)
    return builder