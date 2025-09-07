import time
import socket
import threading
import signal
from typing import Dict, Any, Callable, Optional
from multiprocessing.managers import SharedMemoryManager, BaseManager
from queue import Empty, Queue
import numpy as np
import logging
from .shared_memory_queue import SharedMemoryQueue
from .shared_memory_ring_buffer import SharedMemoryRingBuffer


class PubSubTopic:
    """A single topic that can have multiple publishers and subscribers."""
    
    def __init__(self, shm_manager: SharedMemoryManager, examples: Dict[str, Any], 
                 buffer_size: int = 10, mode: str = 'consumer'):
        """
        Initialize a PubSub topic.
        
        Args:
            shm_manager: Shared memory manager
            examples: Example data structure
            buffer_size: Buffer size for the topic
            mode: 'consumer' for consumer model (data consumed on read) 
                  or 'broadcast' for broadcast model (data preserved for multiple readers)
        """
        self.mode = mode
        
        if mode == 'consumer':
            # Consumer model: uses queue, data is consumed on read
            self.storage = SharedMemoryQueue.create_from_examples(
                shm_manager=shm_manager,
                examples=examples,
                buffer_size=buffer_size
            )
        elif mode == 'broadcast':
            # Broadcast model: uses ring buffer, data is preserved for multiple readers
            self.storage = SharedMemoryRingBuffer.create_from_examples(
                shm_manager=shm_manager,
                examples=examples,
                get_max_k=min(buffer_size, 32),  # Limit max read size
                get_time_budget=0.01,  # 10ms time budget
                put_desired_frequency=60.0  # 60 Hz max frequency
            )
        else:
            raise ValueError(f"Unsupported mode: {mode}. Use 'consumer' or 'broadcast'")
        
        self.subscribers = []
        self.last_data = None
        self.lock = threading.Lock()
    
    def publish(self, data: Dict[str, Any]):
        """Publish data to this topic."""
        try:
            self.storage.put(data)
            self.last_data = data
        except Exception as e:
            if self.mode == 'consumer':
                # If queue is full, remove oldest item and add new one
                # This maintains FIFO order while ensuring new data can be added
                try:
                    # Try to consume one old item to make space
                    self.storage.get(block=False)
                    self.storage.put(data)
                    self.last_data = data
                except Empty:
                    # If somehow queue is empty now, just put the new data
                    self.storage.put(data)
                    self.last_data = data
            else:
                # For broadcast mode, ring buffer handles overwriting automatically
                # Just wait a bit and try again
                time.sleep(0.001)
                self.storage.put(data, wait=False)
                self.last_data = data
    
    def subscribe(self, callback: Optional[Callable] = None, get_latest: bool = True):
        """Subscribe to this topic. Returns latest data if get_latest=True."""
        with self.lock:
            if callback:
                self.subscribers.append(callback)
        
        if get_latest:
            try:
                if self.mode == 'consumer':
                    if not self.storage.empty():
                        return self.storage.get(block=False)
                else:  # broadcast mode
                    if self.storage.count > 0:
                        return self.storage.get()
            except (Empty, Exception):
                return None
        return None
    
    def get_latest(self, block: bool = True, timeout: Optional[float] = None):
        """Get the latest data from this topic."""
        if self.mode == 'consumer':
            return self.storage.get(block=block, timeout=timeout)
        else:  # broadcast mode
            if self.storage.count > 0:
                return self.storage.get()
            elif block:
                # For broadcast mode with blocking, we need to wait for new data
                start_time = time.time()
                while self.storage.count == 0:
                    if timeout and (time.time() - start_time) > timeout:
                        raise Empty()
                    time.sleep(0.001)
                return self.storage.get()
            else:
                raise Empty()


class PubSubManager:
    """Manages multiple topics and handles automatic port management."""
    
    def __init__(self, port: int, authkey: bytes = b'pubsub'):
        self.port = port
        self.authkey = authkey
        self.topics: Dict[str, PubSubTopic] = {}
        self.shm_manager = None
        self.manager = None
        self.server_process = None
        self.is_server = False
        self.publisher_count = 0
        self.subscriber_count = 0
        self.lock = threading.Lock()
        self.heartbeat_thread = None
        self.running = False
        # New: Auto-listener management
        self.listener_threads = {}  # topic_name -> thread
        self.topic_handlers = {}    # topic_name -> handler_function
        self.shutdown_requested = False
    
    def _is_port_available(self, port: int) -> bool:
        """Check if a port is available."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                # Enable port reuse to handle cases where previous process was killed
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind(('127.0.0.1', port))
                return True
            except OSError:
                return False
    
    def _setup_server(self):
        """Setup the server if port is available."""
        if not self._is_port_available(self.port):
            return False
        
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        
        # Custom manager class
        class TopicManager(BaseManager):
            pass
        
        # Register methods
        TopicManager.register('get_topics', callable=lambda: self.topics)
        TopicManager.register('create_topic', callable=self._create_topic)
        TopicManager.register('get_topic', callable=self._get_topic)
        TopicManager.register('register_publisher', callable=self._register_publisher)
        TopicManager.register('register_subscriber', callable=self._register_subscriber)
        TopicManager.register('unregister_publisher', callable=self._unregister_publisher)
        TopicManager.register('unregister_subscriber', callable=self._unregister_subscriber)
        TopicManager.register('get_client_counts', callable=self._get_client_counts)
        
        self.manager = TopicManager(address=('127.0.0.1', self.port), authkey=self.authkey)
        self.server = self.manager.get_server()
        
        # Start server in a separate thread (daemon so it can be killed)
        self.server_thread = threading.Thread(target=self._server_runner, daemon=True)
        self.server_thread.start()
        
        # Start heartbeat monitoring thread
        self.running = True
        self.heartbeat_thread = threading.Thread(target=self._monitor_clients, daemon=True)
        self.heartbeat_thread.start()
        
        self.is_server = True
        logging.info(f"PubSub server started on port {self.port}")
        return True
    
    def _server_runner(self):
        """Run the server with proper exception handling."""
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logging.info("Server interrupted by user")
        except Exception as e:
            logging.error(f"Server error: {e}")
        finally:
            self._shutdown_server()
    
    def _force_setup_server(self):
        """Force setup server even if port seems occupied (for recovery from killed processes)."""
        try:
            self.shm_manager = SharedMemoryManager()
            self.shm_manager.start()
            
            # Custom manager class
            class TopicManager(BaseManager):
                pass
            
            # Register methods
            TopicManager.register('get_topics', callable=lambda: self.topics)
            TopicManager.register('create_topic', callable=self._create_topic)
            TopicManager.register('get_topic', callable=self._get_topic)
            TopicManager.register('register_publisher', callable=self._register_publisher)
            TopicManager.register('register_subscriber', callable=self._register_subscriber)
            TopicManager.register('unregister_publisher', callable=self._unregister_publisher)
            TopicManager.register('unregister_subscriber', callable=self._unregister_subscriber)
            TopicManager.register('get_client_counts', callable=self._get_client_counts)
            
            self.manager = TopicManager(address=('127.0.0.1', self.port), authkey=self.authkey)
            
            # Force bind with SO_REUSEADDR
            import socket
            self.server = self.manager.get_server()
            if hasattr(self.server, 'listener') and hasattr(self.server.listener, '_socket'):
                self.server.listener._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            logging.info(f"PubSub server force-started on port {self.port}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to force setup server: {e}")
            if self.shm_manager:
                try:
                    self.shm_manager.shutdown()
                except:
                    pass
            return False
    
    def _connect_to_server(self):
        """Connect to existing server."""
        class TopicManager(BaseManager):
            pass
        
        TopicManager.register('get_topics')
        TopicManager.register('create_topic')
        TopicManager.register('get_topic')
        TopicManager.register('register_publisher')
        TopicManager.register('register_subscriber')
        TopicManager.register('unregister_publisher')
        TopicManager.register('unregister_subscriber')
        TopicManager.register('get_client_counts')
        
        self.manager = TopicManager(address=('127.0.0.1', self.port), authkey=self.authkey)
        self.manager.connect()
        logging.info(f"Connected to PubSub server on port {self.port}")
    
    def start(self, role: str = 'both'):
        """Start the PubSub manager (server or client).
        
        Args:
            role: 'publisher', 'subscriber', or 'both'
        """
        # First try to setup as server
        if self._setup_server():
            self.is_server = True
            # Register based on role
            if role in ['publisher', 'both']:
                self._register_publisher()
            if role in ['subscriber', 'both']:
                self._register_subscriber()
            return
        
        # If can't be server, try to connect as client with retries
        max_retries = 3
        for attempt in range(max_retries):
            try:
                self._connect_to_server()
                self.is_server = False
                # Register based on role
                if role in ['publisher', 'both']:
                    self.manager.register_publisher()
                if role in ['subscriber', 'both']:
                    self.manager.register_subscriber()
                return
            except ConnectionRefusedError:
                if attempt < max_retries - 1:
                    logging.warning(f"Connection attempt {attempt + 1} failed, retrying...")
                    time.sleep(1)  # Wait 1 second before retry
                else:
                    # Last attempt failed, try to force setup as server
                    logging.warning("All connection attempts failed, trying to force setup as server...")
                    if self._force_setup_server():
                        self.is_server = True
                        # Register based on role
                        if role in ['publisher', 'both']:
                            self._register_publisher()
                        if role in ['subscriber', 'both']:
                            self._register_subscriber()
                        return
                    else:
                        raise ConnectionRefusedError(f"Cannot connect to or create server on port {self.port}")
    
    def stop(self, role: str = 'both'):
        """Stop the PubSub manager and cleanup.
        
        Args:
            role: 'publisher', 'subscriber', or 'both' - what roles to unregister
        """
        try:
            if self.is_server:
                # Unregister locally
                if role in ['publisher', 'both']:
                    self._unregister_publisher()
                if role in ['subscriber', 'both']:
                    self._unregister_subscriber()
            else:
                # Unregister remotely
                if hasattr(self.manager, 'unregister_publisher') and role in ['publisher', 'both']:
                    self.manager.unregister_publisher()
                if hasattr(self.manager, 'unregister_subscriber') and role in ['subscriber', 'both']:
                    self.manager.unregister_subscriber()
        except Exception as e:
            logging.warning(f"Error during client unregistration: {e}")
        
        # Stop auto-listener threads if stopping subscriber role
        if role in ['subscriber', 'both']:
            self.stop_all_listeners()
        
        # Note: Server will shutdown automatically when no clients remain
    
    def _create_topic(self, topic_name: str, examples: Dict[str, Any], 
                     buffer_size: int = 10, mode: str = 'consumer'):
        """Create a new topic."""
        if topic_name not in self.topics:
            if self.is_server:
                self.topics[topic_name] = PubSubTopic(
                    self.shm_manager, examples, buffer_size, mode)
            return True
        return False
    
    def _get_topic(self, topic_name: str) -> Optional[PubSubTopic]:
        """Get an existing topic."""
        return self.topics.get(topic_name)
    
    def _register_publisher(self):
        """Register a publisher."""
        with self.lock:
            self.publisher_count += 1
            logging.info(f"Publisher registered. Total publishers: {self.publisher_count}")
    
    def _register_subscriber(self):
        """Register a subscriber."""
        with self.lock:
            self.subscriber_count += 1
            logging.info(f"Subscriber registered. Total subscribers: {self.subscriber_count}")
    
    def _unregister_publisher(self):
        """Unregister a publisher."""
        with self.lock:
            self.publisher_count = max(0, self.publisher_count - 1)
            logging.info(f"Publisher unregistered. Total publishers: {self.publisher_count}")
    
    def _unregister_subscriber(self):
        """Unregister a subscriber."""
        with self.lock:
            self.subscriber_count = max(0, self.subscriber_count - 1)
            logging.info(f"Subscriber unregistered. Total subscribers: {self.subscriber_count}")
    
    def _get_client_counts(self):
        """Get current client counts."""
        with self.lock:
            return {'publishers': self.publisher_count, 'subscribers': self.subscriber_count}
    
    def _monitor_clients(self):
        """Monitor client counts and shutdown server when no clients remain."""
        while self.running:
            try:
                time.sleep(5)  # Check every 5 seconds
                with self.lock:
                    total_clients = self.publisher_count + self.subscriber_count
                    if total_clients == 0:
                        logging.info("No publishers or subscribers remaining. Shutting down server...")
                        self._shutdown_server()
                        break
            except Exception as e:
                logging.error(f"Error in client monitoring: {e}")
                break
    
    def _shutdown_server(self):
        """Shutdown the server gracefully."""
        self.running = False
        if self.server:
            try:
                self.server.shutdown()
            except:
                pass
        if self.shm_manager:
            try:
                self.shm_manager.shutdown()
            except:
                pass
        logging.info("Server shutdown complete.")
    
    def create_topic(self, topic_name: str, examples: Dict[str, Any], 
                    buffer_size: int = 10, mode: str = 'consumer'):
        """Create a topic (client interface).
        
        Args:
            topic_name: Name of the topic
            examples: Example data structure
            buffer_size: Buffer size for the topic
            mode: 'consumer' for consumer model or 'broadcast' for broadcast model
        """
        if self.is_server:
            # If we are the server, create locally
            return self._create_topic(topic_name, examples, buffer_size, mode)
        elif self.manager:
            # If we are a client, call remote
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return self.manager.create_topic(topic_name, examples, buffer_size, mode)
                except (ConnectionResetError, BrokenPipeError, EOFError, OSError) as e:
                    if attempt == 0:
                        logging.warning(f"Failed to create topic remotely: {e}")
                    # Try to reconnect on first attempt
                    if attempt == 0 and self._reconnect(role='both'):
                        continue
                    elif attempt < max_retries - 1:
                        time.sleep(0.2)
                        continue
                    return False
                except Exception as e:
                    if "server not yet started" in str(e) and attempt < max_retries - 1:
                        time.sleep(0.2)
                        continue
                    elif attempt == 0:
                        logging.warning(f"Failed to create topic remotely: {e}")
                    return False
        return False
    
    def get_topic(self, topic_name: str) -> Optional[PubSubTopic]:
        """Get a topic (client interface)."""
        if self.is_server:
            # If we are the server, get locally
            return self._get_topic(topic_name)
        elif self.manager:
            # If we are a client, call remote
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    return self.manager.get_topic(topic_name)
                except (ConnectionResetError, BrokenPipeError, EOFError, OSError) as e:
                    if attempt == 0:  # Only log on first attempt
                        logging.warning(f"Failed to get topic remotely: {e}")
                    # Try to reconnect on first attempt
                    if attempt == 0 and self._reconnect(role='subscriber'):
                        continue
                    elif attempt < max_retries - 1:
                        time.sleep(0.2)  # Short wait between retries
                        continue
                    return None
                except Exception as e:
                    if "server not yet started" in str(e) and attempt < max_retries - 1:
                        # Server is starting up, wait and retry
                        time.sleep(0.2)
                        continue
                    elif attempt == 0:  # Only log on first attempt
                        logging.warning(f"Failed to get topic remotely: {e}")
                    return None
        return None
    
    def publish(self, topic_name: str, data: Dict[str, Any]):
        """Publish data to a topic with auto-reconnection."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                topic = self.get_topic(topic_name)
                if topic:
                    topic.publish(data)
                    return True
                else:
                    # Topic doesn't exist, try to create it (default to consumer mode)
                    examples = {k: v for k, v in data.items()}
                    if self.create_topic(topic_name, examples, mode='consumer'):
                        topic = self.get_topic(topic_name)
                        if topic:
                            topic.publish(data)
                            return True
            except (ConnectionResetError, BrokenPipeError, EOFError, OSError) as e:
                logging.warning(f"Connection lost during publish attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    # Try to reconnect as publisher
                    if self._reconnect(role='publisher'):
                        continue
                    else:
                        time.sleep(1)
                        continue
            except Exception as e:
                logging.error(f"Unexpected error during publish: {e}")
                break
        
        logging.error(f"Failed to publish to topic '{topic_name}' after {max_retries} attempts")
        return False
    
    def _reconnect(self, role: str = 'both'):
        """Try to reconnect or restart server."""
        logging.info("Attempting to reconnect...")
        try:
            # Reset current connection
            self.manager = None
            if self.shm_manager and self.is_server:
                try:
                    self.shm_manager.shutdown()
                except:
                    pass
                self.shm_manager = None
                self.is_server = False
            
            # Wait a moment for any cleanup to complete
            time.sleep(0.5)
            
            # Try to restart with same role
            self.start(role=role)
            
            # Wait for server to be fully ready
            time.sleep(0.5)
            
            return True
        except Exception as e:
            logging.warning(f"Reconnection failed: {e}")
            return False
    
    def subscribe(self, topic_name: str, callback: Optional[Callable] = None, get_latest: bool = True):
        """Subscribe to a topic."""
        topic = self.get_topic(topic_name)
        if topic:
            return topic.subscribe(callback, get_latest)
        return None
    
    def get_latest(self, topic_name: str, block: bool = True, timeout: Optional[float] = None):
        """Get latest data from a topic with auto-reconnection."""
        try:
            topic = self.get_topic(topic_name)
            if topic:
                return topic.get_latest(block, timeout)
            else:
                # Topic doesn't exist yet, return None
                return None
        except Empty:
            # No data available, this is normal
            return None
        except Exception as e:
            logging.debug(f"Error during get_latest: {e}")
            return None
    
    # High-level convenience methods
    
    def run_publisher(self, topic_name: str, data_generator: Callable, examples: Dict[str, Any], 
                     interval: float = 2.0, buffer_size: int = 5, stats_callback: Optional[Callable] = None):
        """Run a publisher loop with automatic signal handling.
        
        Args:
            topic_name: Name of the topic to publish to
            data_generator: Function that returns data to publish
            examples: Example data structure for topic creation
            interval: Time between publications in seconds
            buffer_size: Topic buffer size
            stats_callback: Optional function to call with (frame_count, start_time, success) for custom stats
        """
        # Setup signal handling
        shutdown_requested = False
        
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            logging.info("Shutdown signal received...")
            shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start as publisher
        self.start(role='publisher')
        
        # Create topic (default to consumer mode for backward compatibility)
        self.create_topic(topic_name, examples, buffer_size, mode='consumer')
        
        try:
            frame_count = 0
            start_time = time.time()
            
            while not shutdown_requested:
                # Generate and publish data
                data = data_generator()
                success = self.publish(topic_name, data)
                
                if success:
                    frame_count += 1
                else:
                    logging.warning("Failed to publish data")
                
                # Call stats callback if provided
                if stats_callback:
                    stats_callback(frame_count, start_time, success)
                
                # Check for shutdown before sleep
                if shutdown_requested:
                    break
                    
                # Only sleep if interval > 0
                if interval > 0:
                    time.sleep(interval)
                
        except KeyboardInterrupt:
            logging.info("Publisher shutting down by user.")
        finally:
            self.stop(role='publisher')
            logging.info("Publisher has shut down.")
    
    def run_subscriber(self, request_topic: str, response_topic: str, 
                      request_examples: Dict[str, Any], response_examples: Dict[str, Any],
                      request_handler: Optional[Callable] = None, buffer_size: int = 5):
        """Run a subscriber loop with automatic signal handling.
        
        Args:
            request_topic: Name of the request topic to subscribe to
            response_topic: Name of the response topic to subscribe to  
            request_examples: Example data structure for request topic
            response_examples: Example data structure for response topic
            request_handler: Optional function to process requests
            buffer_size: Topic buffer size
        """
        # Setup signal handling
        shutdown_requested = False
        
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            logging.info("Shutdown signal received...")
            shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Start as subscriber
        self.start(role='subscriber')
        
        # Create topics (default to consumer mode for backward compatibility)
        self.create_topic(request_topic, request_examples, buffer_size, mode='consumer')
        self.create_topic(response_topic, response_examples, buffer_size, mode='consumer')
        
        try:
            request_count = 0
            response_count = 0
            start_time = time.time()
            
            while not shutdown_requested:
                try:
                    # Check for new requests
                    req_data = self.get_latest(request_topic, block=False)
                    if req_data is not None:
                        request_count += 1
                        
                        # Process request if handler provided
                        if request_handler:
                            response = request_handler(req_data)
                            if response:
                                self.publish(response_topic, response)
                    
                    # Check for new responses
                    res_data = self.get_latest(response_topic, block=False)
                    if res_data is not None:
                        response_count += 1
                    
                    # Print stats every 5 seconds for better visibility
                    current_time = time.time()
                    if current_time - start_time >= 5:
                        elapsed = current_time - start_time
                        req_fps = request_count / elapsed
                        res_fps = response_count / elapsed
                        logging.info(f"Stats - Requests: {request_count} ({req_fps:.2f}/s), Responses: {response_count} ({res_fps:.2f}/s)")
                        
                        # Reset counters
                        request_count = 0
                        response_count = 0
                        start_time = current_time
                    
                    # Reduce sleep time for higher throughput
                    time.sleep(0.001)  # 1ms instead of 100ms
                    
                except Exception as e:
                    logging.debug(f"No new data available: {e}")
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logging.info("Subscriber shutting down by user.")
        finally:
            self.stop(role='subscriber')
            logging.info("Subscriber has shut down.")
    
    def setup_subscriber(self, topics_config: Dict[str, Dict[str, Any]]):
        """Setup subscriber with topic configurations.
        
        Args:
            topics_config: Dict of {topic_name: {'examples': {...}, 'buffer_size': int}}
        """
        # Start as subscriber if not already started
        if not hasattr(self, 'manager') or self.manager is None:
            self.start(role='subscriber')
        
        # Store topic configurations
        self._subscriber_topics = topics_config
        
        # Create all topics
        for topic_name, config in topics_config.items():
            examples = config['examples']
            buffer_size = config.get('buffer_size', 5)
            mode = config.get('mode', 'consumer')  # Default to consumer mode
            self.create_topic(topic_name, examples, buffer_size, mode)
    
    def get_latest_data(self, topic_name: str):
        """Simple interface to get the latest data from a configured topic.
        
        Args:
            topic_name: Name of the topic to get data from
            
        Returns:
            Latest data from the topic or None if no data available
        """
        # Check if topic is configured
        if not hasattr(self, '_subscriber_topics') or topic_name not in self._subscriber_topics:
            raise ValueError(f"Topic '{topic_name}' not configured. Call setup_subscriber() first.")
        
        # Get latest data
        return self.get_latest(topic_name, block=False)
    
    def run_data_listener(self, topic_name: str, data_handler: Callable, 
                         check_interval: float = 0.001):
        """Run a listener that calls a handler function when new data arrives.
        
        Args:
            topic_name: Name of the topic to listen to (must be configured first)
            data_handler: Function to call with new data (data) -> None
            check_interval: How often to check for new data (seconds)
        """
        # Setup signal handling
        shutdown_requested = False
        
        def signal_handler(signum, frame):
            nonlocal shutdown_requested
            logging.info("Shutdown signal received...")
            shutdown_requested = True
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Check if topic is configured
        if not hasattr(self, '_subscriber_topics') or topic_name not in self._subscriber_topics:
            raise ValueError(f"Topic '{topic_name}' not configured. Call setup_subscriber() first.")
        
        try:
            data_count = 0
            start_time = time.time()
            last_data = None
            
            while not shutdown_requested:
                try:
                    # Get latest data
                    current_data = self.get_latest(topic_name, block=False)
                    
                    # Check if we have new data (simple comparison)
                    if current_data is not None and current_data != last_data:
                        data_count += 1
                        last_data = current_data
                        
                        # Call the handler function
                        try:
                            data_handler(current_data)
                        except Exception as e:
                            logging.error(f"Error in data handler: {e}")
                        
                        # Print stats every 100 items
                        if data_count % 100 == 0:
                            elapsed = time.time() - start_time
                            fps = data_count / elapsed
                            logging.info(f"Processed {data_count} items, rate: {fps:.1f}/s")
                    
                    # Sleep for the specified interval
                    if check_interval > 0:
                        time.sleep(check_interval)
                        
                except Exception as e:
                    logging.debug(f"Error in listener loop: {e}")
                    time.sleep(0.01)
                    
        except KeyboardInterrupt:
            logging.info("Data listener shutting down by user.")
        finally:
            self.stop(role='subscriber')
            logging.info("Data listener has shut down.")
    
    def register_topic_handler(self, topic_name: str, handler: Callable, check_interval: float = 0.001):
        """Register a handler function for a topic. Auto-creates and manages listener thread."""
        if topic_name in self.topic_handlers:
            logging.warning(f"Handler for topic '{topic_name}' already registered. Replacing...")
            self.unregister_topic_handler(topic_name)
        
        # Store the handler
        self.topic_handlers[topic_name] = handler
        
        # Create and start listener thread
        def listener_worker():
            """Worker function for the listener thread."""
            logging.info(f"Auto-listener started for topic '{topic_name}'")
            data_count = 0
            start_time = time.time()
            
            try:
                while not self.shutdown_requested:
                    try:
                        # Get topic (with reconnection logic)
                        topic = self.get_topic(topic_name)
                        if topic is None:
                            time.sleep(0.1)
                            continue
                        
                        # Check for new data
                        data = topic.get_latest(block=False)
                        if data is not None:
                            try:
                                # Call the registered handler
                                handler(data)
                                data_count += 1
                                
                                # Log stats every 100 items
                                if data_count % 100 == 0:
                                    elapsed = time.time() - start_time
                                    if elapsed > 0:
                                        fps = data_count / elapsed
                                        logging.debug(f"Topic '{topic_name}': processed {data_count} items, rate: {fps:.1f}/s")
                            except Exception as e:
                                logging.error(f"Error in handler for topic '{topic_name}': {e}")
                        
                        # Sleep for the specified interval
                        if check_interval > 0:
                            time.sleep(check_interval)
                            
                    except Exception as e:
                        logging.debug(f"Error in auto-listener for topic '{topic_name}': {e}")
                        time.sleep(0.01)
                        
            except Exception as e:
                logging.error(f"Fatal error in auto-listener for topic '{topic_name}': {e}")
            finally:
                logging.info(f"Auto-listener stopped for topic '{topic_name}'")
        
        # Start the listener thread
        thread = threading.Thread(target=listener_worker, daemon=True)
        thread.start()
        self.listener_threads[topic_name] = thread
        
        logging.info(f"Registered auto-listener for topic '{topic_name}'")
    
    def unregister_topic_handler(self, topic_name: str):
        """Unregister a handler and stop its listener thread."""
        if topic_name in self.topic_handlers:
            del self.topic_handlers[topic_name]
            logging.info(f"Unregistered handler for topic '{topic_name}'")
        
        if topic_name in self.listener_threads:
            # The thread will stop when shutdown_requested is set
            thread = self.listener_threads.pop(topic_name)
            logging.info(f"Stopped auto-listener thread for topic '{topic_name}'")
    
    def stop_all_listeners(self):
        """Stop all auto-listener threads."""
        self.shutdown_requested = True
        
        # Wait for all threads to finish
        for topic_name, thread in self.listener_threads.items():
            if thread.is_alive():
                logging.debug(f"Waiting for auto-listener thread '{topic_name}' to stop...")
                thread.join(timeout=1.0)
        
        self.listener_threads.clear()
        self.topic_handlers.clear()
        logging.info("All auto-listener threads stopped.")