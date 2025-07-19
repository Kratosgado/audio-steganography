import numpy as np
import librosa
from core_modules.embedding_module import EmbeddingModule
from core_modules.preprocessor import AudioPreprocessor
from core_modules.config import cfg


NORM_NUM = 32768


class SpreadSpectrum(EmbeddingModule):
    def __init__(self):
        """
        Initialize the Spread Spectrum steganography system

        Parameters:
          carrier_freq (int): Carrier frequency in Hz for embedding
          chip_rate (int): How many samples per bit (spreading factor)
          snr (int): Desired Signal to noise ratio in dB for embedding
        """
        # Parameters for Gold sequence generation (example values)
        self.taps1 = [5, 2]
        self.taps2 = [5, 4, 2, 1]  # Must be a preferred pair with taps1
        self.seed1 = 0b11111
        self.seed2 = 0b10101

        self.action_ranges = {
            "carrier_freq": (5000, 15000),
            "chip_rate": (50, 200),
            "snr": (10, 30),
        }

    def _scale_action(self, normalized_val, low, high):
        """Scale from [-1, 1] to [low, high]"""
        return low + (normalized_val + 1) * (high - low) / 2

    def set_parameters(self, action):
        """
        Parameters(action):
          carrier_freq (int): Carrier frequency in Hz for embedding
          chip_rate (int): How many samples per bit (spreading factor)
          snr (int): Desired Signal to noise ratio in dB for embedding
        """
        # Scale normalized actions to original ranges
        self.carrier_freq = int(
            self._scale_action(action[0], *self.action_ranges["carrier_freq"])
        )
        self.chip_rate = int(
            self._scale_action(action[1], *self.action_ranges["chip_rate"])
        )
        self.snr_db = int(self._scale_action(action[2], *self.action_ranges["snr"]))
        return self.carrier_freq, self.chip_rate, self.snr_db

    def _generate_m_sequence(self, taps, length, initial_state):
        """Generate an m-sequence."""
        lfsr = initial_state
        seq = np.zeros(length)
        # mask = sum(1 << (t - 1) for t in taps) # Mask is not used in this implementation

        for i in range(length):
            seq[i] = 1 if (lfsr & 1) else -1  # Convert to bipolar (-1, 1)
            feedback = 0
            for tap in taps:
                feedback ^= (lfsr >> (tap - 1)) & 1
            # Adjust the shift based on the number of bits in the initial state
            lfsr = (lfsr >> 1) | (
                feedback << (len(bin(initial_state)) - 3)
            )  # Assuming initial_state is not 0

        return seq

    def _generate_spreading_code(self, length):
        """Generate a Gold sequence."""
        # The lengths of the m-sequences must be the same
        # and derived from the same primitive polynomial.
        # The taps define the primitive polynomials.
        # The seeds are the initial states of the LFSRs.

        m_seq1 = self._generate_m_sequence(self.taps1, length, self.seed1)
        m_seq2 = self._generate_m_sequence(self.taps2, length, self.seed2)

        # XOR the two m-sequences
        gold_seq = m_seq1 * m_seq2  # For bipolar sequences, XOR is multiplication

        return gold_seq

    def _int_to_bits(self, decimal_value, num_bits):
        """Convert a decimal value to a binary string of a fixed number of bits."""
        binary_string = bin(decimal_value)[2:].zfill(num_bits)
        return [int(bit) for bit in binary_string]

    def _bits_to_int(self, binary_string):
        """Convert a binary string to a decimal value."""
        return int(binary_string, 2)

    def _embed_bits_lsb(self, audio, bits_to_embed, start_sample=0):
        """Embed a sequence of bits into the least significant bits of audio samples."""
        audio_int = (audio * NORM_NUM).astype(np.int16)
        
        if len(bits_to_embed) > len(audio_int):
            raise ValueError("Not enough audio samples to embed all bits.")

        # Create a copy to modify
        audio_int_modified = audio_int.copy()

        # Embed the bits starting from start_sample
        for i, bit in enumerate(bits_to_embed):
            if start_sample + i < len(audio_int_modified):
                # Replace the LSB of each 16-bit sample
                sample_index = start_sample + i
                # Clear the LSB and set it to the new bit
                audio_int_modified[sample_index] = (audio_int_modified[sample_index] & 0xFE) | bit

        return audio_int_modified.astype(np.float32) / NORM_NUM

    def _extract_bits_lsb(self, audio, num_bits_to_extract, start_sample=0):
        """Extract a sequence of bits from the least significant bits of audio samples."""
        audio_int = (audio * NORM_NUM).astype(np.int16)

        if num_bits_to_extract + start_sample > len(audio_int):
            raise ValueError("Cannot extract more bits than available LSBs.")

        extracted_bits = []
        for i in range(num_bits_to_extract):
            sample_index = start_sample + i
            # Extract the LSB
            bit = audio_int[sample_index] & 1
            extracted_bits.append(bit)
            
        return extracted_bits

    def embed(self, original_audio, msg_bits: np.ndarray, action, **kwargs):
        """
        Embed a message into an audio file using spread spectrum with LSB hiding of parameters.

        Parameters:
          original_audio (ndarray): Original audio data
          msg_bits (ndarray): Message bits to embed
          action (tuple): Parameters for embedding (carrier_freq, chip_rate, snr_db)
        """
        # Set parameters from action
        carrier_freq, chip_rate, snr_db = self.set_parameters(action)
        
        # Calculate message length in bits
        message_length = len(msg_bits)
        
        # Convert parameters to binary strings for LSB embedding
        carrier_freq_bits = self._int_to_bits(carrier_freq, 16)
        chip_rate_bits = self._int_to_bits(chip_rate, 8)
        snr_db_bits = self._int_to_bits(snr_db, 8)
        message_length_bits = self._int_to_bits(message_length, 16)
        
        # Concatenate all parameter bits
        param_bits = carrier_freq_bits + chip_rate_bits + snr_db_bits + message_length_bits
        total_param_bits = len(param_bits)
        
        print(f"Embedding parameters: carrier_freq={carrier_freq}, chip_rate={chip_rate}, snr_db={snr_db}, message_length={message_length}")
        print(f"Total parameter bits: {total_param_bits}")
                
        # Convert message bits to bipolar (-1, 1)
        msg_bits_bipolar = msg_bits * 2 - 1
        
        # Generate spreading codes
        code_length = len(msg_bits_bipolar) * chip_rate
        spreading_code = self._generate_spreading_code(code_length)
        
        # Create the spread message signal
        spread_message = np.repeat(msg_bits_bipolar, chip_rate) * spreading_code
        
        # Create carrier signal (sine wave at carrier frequency)
        t = np.arange(len(spread_message)) / cfg.SAMPLE_RATE
        carrier = np.sin(2 * np.pi * carrier_freq * t)
        
        # Modulate the message onto the carrier
        modulated = spread_message * carrier
        
        # Adjust the signal power based on desired SNR
        signal_power = np.var(original_audio)
        message_power = np.var(modulated)
        desired_message_power = signal_power / (10 ** (snr_db / 10))
        scaling_factor = np.sqrt(desired_message_power / message_power)
        modulated = modulated * scaling_factor
        
        stego_audio = original_audio.copy()
        # Pad or truncate the modulated signal to match audio length
        if len(modulated) < len(stego_audio):
            modulated = np.pad(
                modulated, (0, len(stego_audio) - len(modulated)), "constant"
            )
        else:
            modulated = modulated[: len(stego_audio)]
        
        # Add the modulated signal to the audio (starting after parameter bits)
        # We'll add it to the entire audio since the LSB embedding is minimal
        stego_audio = stego_audio + modulated
        stego_audio = self._embed_bits_lsb(stego_audio, param_bits, start_sample=0)

        
        return stego_audio

    def extract(self, stego_audio, message_length=None, original_audio_path=None, **kwargs):
        """
        Extract a hidden message from a stego audio file.

        Parameters:
            stego_audio (ndarray): Stego audio data
            message_length (int): Optional length of the hidden message in bits
            original_audio_path (str): Optional path to original audio for comparison
        """
        # First, extract the parameter bits using LSB
        num_carrier_freq_bits = 16
        num_chip_rate_bits = 8
        num_snr_db_bits = 8
        num_message_length_bits = 16
        total_param_bits = num_carrier_freq_bits + num_chip_rate_bits + num_snr_db_bits + num_message_length_bits
        
        print(f"Extracting {total_param_bits} parameter bits from LSB...")
        
        extracted_param_bits = self._extract_bits_lsb(stego_audio, total_param_bits, start_sample=0)
        
        # Convert extracted parameter bits back to decimal values
        carrier_freq_bits_str = "".join(
            str(bit) for bit in extracted_param_bits[:num_carrier_freq_bits]
        )
        chip_rate_bits_str = "".join(
            str(bit)
            for bit in extracted_param_bits[
                num_carrier_freq_bits : num_carrier_freq_bits + num_chip_rate_bits
            ]
        )
        snr_db_bits_str = "".join(
            str(bit)
            for bit in extracted_param_bits[
                num_carrier_freq_bits + num_chip_rate_bits : num_carrier_freq_bits + num_chip_rate_bits + num_snr_db_bits
            ]
        )
        message_length_bits_str = "".join(
            str(bit)
            for bit in extracted_param_bits[
                num_carrier_freq_bits + num_chip_rate_bits + num_snr_db_bits :
            ]
        )
        
        extracted_carrier_freq = self._bits_to_int(carrier_freq_bits_str)
        extracted_chip_rate = self._bits_to_int(chip_rate_bits_str)
        extracted_snr_db = self._bits_to_int(snr_db_bits_str)
        extracted_message_length = self._bits_to_int(message_length_bits_str)
        
        print(f"Extracted parameters: carrier_freq={extracted_carrier_freq}, chip_rate={extracted_chip_rate}, snr_db={extracted_snr_db}, message_length={extracted_message_length}")
        
        # Use the extracted parameters for message extraction
        self.carrier_freq = int(extracted_carrier_freq)
        self.chip_rate = int(extracted_chip_rate)
        self.snr_db = int(extracted_snr_db)
        
        # Use extracted message length if not provided
        if message_length is None:
            message_length = extracted_message_length
        
        # If original audio is provided, subtract it to get just the message
        if original_audio_path:
            y_original, _ = librosa.load(original_audio_path, sr=cfg.SAMPLE_RATE)
            y_diff = stego_audio - y_original
        else:
            y_diff = stego_audio
        
        # Generate the same spreading code used in embedding
        code_length = message_length * self.chip_rate
        spreading_code = self._generate_spreading_code(code_length)
        
        # Create carrier signal
        t = np.arange(len(spreading_code)) / cfg.SAMPLE_RATE
        carrier = np.sin(2 * np.pi * self.carrier_freq * t)
        
        # Pad or truncate the carrier to match the difference signal
        if len(carrier) < len(y_diff):
            carrier = np.pad(carrier, (0, len(y_diff) - len(carrier)), "constant")
        else:
            carrier = carrier[: len(y_diff)]
        
        # Demodulate the signal
        demodulated = y_diff * carrier
        
        # Correlate with spreading code to extract bits
        extracted_bits = []
        for i in range(message_length):
            start = i * self.chip_rate
            end = start + self.chip_rate
            if end > len(demodulated):
                break
            segment = demodulated[start:end]
            code_segment = spreading_code[start:end]
            
            # Calculate correlation
            correlation = np.sum(segment * code_segment)
            extracted_bits.append(1 if correlation > 0 else 0)
        
        print(f"Extracted {len(extracted_bits)} bits from spread spectrum")
        return extracted_bits
