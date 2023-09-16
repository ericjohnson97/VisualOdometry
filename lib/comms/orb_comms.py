import struct
import socket
import cv2
import numpy as np
from typing import List, Tuple



class ORBcomms:

    def __init__(self, mode="send", ip="127.0.0.1", port=7000) -> None:

        if mode == "send":
            self.send = True
            self.recv = False
            self.create_sender(ip, port)
        elif mode == "recv":
            self.send = False
            self.recv = True
            self.create_receiver(ip, port)
        else:
            raise ValueError("Mode must be 'send' or 'recv'")
    
    def create_receiver(self, ip, port):
        """
        Creates a UDP socket
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((ip, port))

    def create_sender(self, dest_ip, dest_port):
        """
        Creates a UDP socket
        """
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.dest = (dest_ip, dest_port)
    
    def receive_orb_data(self):
        """
        Receives the keypoints and descriptors from the source address.

        Returns:
            list, np.ndarray: A list of keypoints and a numpy array of descriptors.
        """
        if not self.recv:
            raise ValueError("This instance is not configured to receive data.")
        data, addr = self.sock.recvfrom(65536)
        return self.deserialize_input(data)

    def send_orb_data(self, keypoints, descriptors):
        """
        Sends the keypoints and descriptors to the destination address.

        Args:
            keypoints (list): A list of keypoints with various attributes.
            descriptors (np.ndarray): A numpy array containing concatenated vectors of descriptors.
        """
        if not self.send:
            raise ValueError("This instance is not configured to send data.")
        data = self._serialize_output(keypoints, descriptors)
        self.sock.sendto(data, self.dest)


    def _serialize_output(self, keypoints: List[cv2.KeyPoint], descriptors: np.ndarray) -> bytes:
        """
        Serializes the output of the cv.ORB/detectAndCompute method to a binary format.

        Args:
            keypoints: A list of keypoints with various attributes.
            descriptors (np.ndarray): A numpy array containing concatenated vectors of descriptors.

        Returns:
            bytes: A binary representation of the keypoints and descriptors.
        """
        serialized_keypoints = []
        for kp in keypoints:
            # Packaging individual keypoint attributes into binary format
            serialized_keypoints.append(struct.pack('2f f f f i i', kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave, kp.class_id))
        
        serialized_descriptors = descriptors.tobytes()
        # Display the size of the serialized descriptor data for debugging
        print(f"Serialized descriptor data size: {len(serialized_descriptors)}")
        keypoints_len = len(serialized_keypoints)
        descriptors_len = descriptors.size  # Total size in bytes
        descriptors_shape = descriptors.shape  # Capturing the original shape of the descriptors array
        print(f"Original descriptor shape: {descriptors.shape}")
        # Packaging all serialized data along with the shape and size information into a binary format
        keypoint_data_size = struct.calcsize('2f f f f i i')  # Calculating the size of a single keypoint record
        descriptor_data_size = descriptors_len * 8  # Assuming descriptors are stored as 64-bit (8 bytes) values

        data = struct.pack(f'i {keypoint_data_size * keypoints_len}s {descriptor_data_size}s ii', keypoints_len, b''.join(serialized_keypoints), serialized_descriptors, *descriptors_shape)
        print(f"Serialized data size: {len(data)}")

        return data




    def deserialize_input(self, data: bytes) -> Tuple[List[cv2.KeyPoint], np.ndarray]:
        """
        Deserializes a binary string to the data structures used as the output of the cv.ORB/detectAndCompute method.

        Args:
            data (bytes): A binary representation of the keypoints and descriptors.

        Returns:
            Tuple[List[cv2.KeyPoint], np.ndarray]: A list of keypoints and a numpy array of descriptors.
        """
        offset = 0

        # Unpacking the number of keypoints from the binary data
        keypoints_len, = struct.unpack_from('i', data, offset)
        offset += struct.calcsize('i')  # Incrementing the offset to point to the start of keypoints data

        keypoint_record_size = struct.calcsize('2f f f f i i')
        keypoints_data = data[offset:offset+keypoints_len*keypoint_record_size]
        offset += keypoints_len*keypoint_record_size  # Incrementing the offset to point to the start of descriptor data

        keypoints = []
        print(f"keypoints_len: {keypoints_len}")
        for i in range(keypoints_len):
            # Extracting and unpacking individual keypoint data to create cv2.KeyPoint objects
            kp_data = keypoints_data[i*keypoint_record_size:(i+1)*keypoint_record_size]
            pt_x, pt_y, size, angle, response, octave, class_id = struct.unpack('2f f f f i i', kp_data)
            keypoints.append(cv2.KeyPoint(x=pt_x, y=pt_y, size=size, angle=angle, response=response, octave=octave, class_id=class_id))

        print(f"len of keypoints: {len(keypoints)}")

        # Extracting descriptor data and then unpacking shape information from the end of the binary data
        descriptor_shape = struct.unpack_from('ii', data, -8)  # Read the shape information (num_descriptors and descriptor_length)
        expected_descriptor_data_size = np.prod(descriptor_shape) * 1  # uint8 data type (1 byte per element)
        
        descriptors_data = data[offset:offset+expected_descriptor_data_size]  # Extracting based on the byte size
        
        print(f"Expected shape: {descriptor_shape}, actual data size: {len(descriptors_data)}")
        
        # Check if the size of the descriptor data matches the expected size based on the shape information
        assert len(descriptors_data) == expected_descriptor_data_size, "Data size does not match expected shape"
        
        # Creating a numpy array from descriptor data with the correct shape
        descriptors = np.frombuffer(descriptors_data, dtype=np.uint8).reshape(descriptor_shape)
        
        print(f"types: {type(descriptors)}, {type(keypoints)}")

        return keypoints, descriptors
