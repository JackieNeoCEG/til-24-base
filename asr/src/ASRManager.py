import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from io import BytesIO

class ASRManager:
    def __init__(self, model_path="./workspace/fine_tuned_whisper"):
        # Load the Whisper processor and model
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = WhisperForConditionalGeneration.from_pretrained(model_path)
        
    def transcribe(self, audio_bytes: bytes) -> str:
        try:
            # Load audio from bytes using torchaudio
            audio_tensor, sample_rate = torchaudio.load(BytesIO(audio_bytes))
            
            # Ensure audio is 16 kHz
            if sample_rate != 16000:
                audio_tensor = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)(audio_tensor)
                sample_rate = 16000
            
            # Handle multi-channel audio by averaging channels to mono
            if audio_tensor.dim() > 1:
                audio_tensor = torch.mean(audio_tensor, dim=0)
            
            # Convert tensor to numpy array as Whisper expects numpy input
            audio_np = audio_tensor.numpy().flatten()
            
            # Preprocess audio and prepare for the model
            inputs = self.processor(audio_np, sampling_rate=16000, return_tensors="pt")
            
            # Perform transcription
            with torch.no_grad():
                predicted_ids = self.model.generate(inputs.input_features, max_length=448)
                
            # Decode the generated IDs to text
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
            
            return transcription[0].strip()
        except Exception as e:
            print(f"Error in transcription: {e}")
            return ""

