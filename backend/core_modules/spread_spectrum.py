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

    def _decimal_to_binary(self, decimal_value, num_bits):
        """Convert a decimal value to a binary string of a fixed number of bits."""
        return bin(decimal_value)[2:].zfill(num_bits)

    def _binary_to_decimal(self, binary_string):
        """Convert a binary string to a decimal value."""
        return int(binary_string, 2)

    def _embed_bits_lsb(self, audio, bits_to_embed):
        """Embed a sequence of bits into the least significant bits of audio samples."""
        audio_int = (audio * NORM_NUM).astype(np.int16)
        audio_bits = np.unpackbits(audio_int.view(np.uint8))

        if len(bits_to_embed) > len(audio_bits):
            raise ValueError("Not enough audio samples to embed all bits.")

        # Replace LSBs of audio_bits with bits_to_embed
        # This is a simplified approach, typically you'd embed into specific bytes/bits
        # based on the audio format and desired imperceptibility.
        # Here, we'll replace the last bit of each 16-bit audio sample (2 bytes)
        num_audio_bytes = len(audio_int) * 2
        num_bits_per_sample = 16
        lsb_indices = np.arange(num_audio_bytes) * num_bits_per_sample + (
            num_bits_per_sample - 1
        )

        # Ensure we don't exceed the available LSBs
        if len(bits_to_embed) > len(lsb_indices):
            raise ValueError("Not enough LSBs in audio to embed all bits.")

        # Create a copy to modify
        audio_bits_modified = audio_bits.copy()

        # Embed the bits
        for i, bit in enumerate(bits_to_embed):
            audio_bits_modified[lsb_indices[i]] = bit

        # Pack the modified bits back into int16
        audio_int_modified = np.packbits(audio_bits_modified).view(np.int16)
        return audio_int_modified.astype(np.float32) / NORM_NUM

    def _extract_bits_lsb(self, audio, num_bits_to_extract):
        """Extract a sequence of bits from the least significant bits of audio samples."""
        audio_int = (audio * NORM_NUM).astype(np.int16)
        audio_bits = np.unpackbits(audio_int.view(np.uint8))

        num_audio_bytes = len(audio_int) * 2
        num_bits_per_sample = 16
        lsb_indices = np.arange(num_audio_bytes) * num_bits_per_sample + (
            num_bits_per_sample - 1
        )

        if num_bits_to_extract > len(lsb_indices):
            raise ValueError("Cannot extract more bits than available LSBs.")

        extracted_bits = [
            audio_bits[lsb_indices[i]] for i in range(num_bits_to_extract)
        ]
        return extracted_bits

    def embed(self, original_audio, msg_bits: np.ndarray, action, **kwargs):
        """
        Embed a message into an audio file.

        Parameters:
          original_audio (str): Path to the audio file.
          message (str): The message to be embedded.
          output_path (str): Path to save the embedded audio file.
          action (tuple): Parameters for embedding (carrier_freq, chip_rate, snr_db)
        """
        carrier_freq, chip_rate, snr_db = self.set_parameters(action)
        # Convert parameters to binary strings
        # Assuming a fixed number of bits for each parameter for simplicity
        carrier_freq_bits = self._decimal_to_binary(
            carrier_freq, 16
        )  # Example: 16 bits for carrier freq
        chip_rate_bits = self._decimal_to_binary(
            chip_rate, 8
        )  # Example: 8 bits for chip rate
        snr_db_bits = self._decimal_to_binary(
            int(snr_db), 8
        )  # Example: 8 bits for SNR dB

        # Concatenate parameter bits
        param_bits = [
            int(bit) for bit in carrier_freq_bits + chip_rate_bits + snr_db_bits
        ]

        # Embed parameter bits using LSB
        stego_audio = self._embed_bits_lsb(original_audio.copy(), param_bits)

        msg_bits = msg_bits * 2 - 1  # convert to bipoloar (-1, 1)

        # generate spreading codes
        code_length = len(msg_bits) * chip_rate
        spreading_code = self._generate_spreading_code(code_length)

        # create the spread message signal
        spread_message = np.repeat(msg_bits, chip_rate) * spreading_code

        # create carrier signal (sine wave at carrier frequency)
        t = np.arange(len(spread_message)) / cfg.SAMPLE_RATE
        carrier = np.sin(2 * np.pi * carrier_freq * t)

        # modulate the message onto the carrier
        modulated = spread_message * carrier

        # adjust the signal power based on desired snr
        signal_power = np.var(original_audio)
        message_power = np.var(modulated)
        desired_message_power = signal_power / (10 ** (snr_db / 10))
        scaling_factor = np.sqrt(desired_message_power / message_power)
        modulated = modulated * scaling_factor

        # pad or truncate the modulated signal to match audio length
        if len(modulated) < len(stego_audio):
            modulated = np.pad(
                modulated, (0, len(stego_audio) - len(modulated)), "constant"
            )
        else:
            modulated = modulated[: len(stego_audio)]

        # embed the message into the audio (add to the audio with embedded parameters)
        stego_audio = stego_audio + modulated
        return stego_audio

    def extract(self, stego_audio, message_length, original_audio_path=None, **kwargs):
        """
        Extract a hidden message from a stego audio file.

        Parameters:
            stego_audio (nd.ndarray): Path to the stego audio file
            message_length (int): Length of the hidden message in bits
            original_audio_path (str): Optional path to original audio for comparison
        """
        # First, extract the parameter bits using LSB
        # Assuming the same number of bits used for embedding each parameter
        num_carrier_freq_bits = 16
        num_chip_rate_bits = 8
        num_snr_db_bits = 8
        total_param_bits = num_carrier_freq_bits + num_chip_rate_bits + num_snr_db_bits

        extracted_param_bits = self._extract_bits_lsb(stego_audio, total_param_bits)

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
                num_carrier_freq_bits + num_chip_rate_bits :
            ]
        )

        extracted_carrier_freq = self._binary_to_decimal(carrier_freq_bits_str)
        extracted_chip_rate = self._binary_to_decimal(chip_rate_bits_str)
        extracted_snr_db = self._binary_to_decimal(snr_db_bits_str)

        # Now use the extracted parameters for message extraction
        self.carrier_freq = int(extracted_carrier_freq)
        self.chip_rate = int(extracted_chip_rate)
        self.snr_db = int(extracted_snr_db)

        # if original audio is provided, subtract it to get just the message
        if original_audio_path:
            y_original, _ = librosa.load(original_audio_path, sr=cfg.SAMPLE_RATE)
            y_diff = stego_audio - y_original
        else:
            y_diff = stego_audio

        # generate the same spreading code used in embedding
        code_length = message_length * 8 * self.chip_rate  # 8 bits per character
        # spreading_code = self._generate_spreading_code(code_length, seed=42) # Old simplified code
        spreading_code = self._generate_spreading_code(code_length)

        # create carrier signal
        t = np.arange(len(spreading_code)) / cfg.SAMPLE_RATE
        carrier = np.sin(2 * np.pi * self.carrier_freq * t)

        # pad or truncate the carrier to match the difference signal
        if len(carrier) < len(y_diff):
            carrier = np.pad(carrier, (0, len(y_diff) - len(carrier)), "constant")
        else:
            carrier = carrier[: len(y_diff)]

        # demodulate the signal
        demodulated = y_diff * carrier

        # correlate with spreading code to extract bits
        extracted_bits: list[int] = []
        for i in range(message_length * 8):
            start = i * self.chip_rate
            end = start + self.chip_rate
            if end > len(demodulated):
                break
            segment = demodulated[start:end]
            code_segment = spreading_code[start:end]

            # calculate correlation
            correlation = np.sum(segment * code_segment)
            extracted_bits.append(1 if correlation > 0 else 0)

        # convert bits to string
        return extracted_bits
