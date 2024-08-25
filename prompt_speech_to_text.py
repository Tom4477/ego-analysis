import groq
from hume import HumeAI
import time
import asyncio

class SpeechToTextAnalyzer:
    SYSTEM_PROMPT = """
    You are an advanced AI assistant specializing in analyzing egocentric audio data to understand everyday activities and emotions. Your task is to process transcripts and emotion data to provide insightful analysis of a person's emotional state and speech content.

    Guidelines:
    1. Analyze the transcript for key topics and sentiment.
    2. Interpret the emotion data in the context of the speech content.
    3. Provide a concise summary of the person's emotional state and the main points of their speech.
    4. Be objective and avoid personal interpretations.
    5. Do not mention AI, language models, or the analysis process.

    Your analysis should be clear, concise, and focused on the person's experience.
    """

    def __init__(self, groq_api_key, hume_api_key):
        self.groq_client = groq.Groq(api_key=groq_api_key)
        self.hume_client = HumeAI(api_key=hume_api_key)

    async def analyze_audio(self, audio_file_path, timestamp):
        transcript = await self._transcribe_audio(audio_file_path)
        emotion_data = await self._analyze_emotion(audio_file_path)
        summary = await self._generate_summary(transcript, emotion_data, timestamp)

        return {
            "timestamp": timestamp,
            "transcript": transcript,
            "emotion": emotion_data,
            "summary": summary
        }

    async def _transcribe_audio(self, audio_file_path):
        with open(audio_file_path, "rb") as file:
            translation = self.groq_client.audio.translations.create(
                file=(audio_file_path, file.read()),
                model="whisper-large-v3",
                prompt="Transcribe the following audio from a first-person perspective video",
                response_format="text",
                temperature=0.0
            )
        return translation.text

    async def _analyze_emotion(self, audio_file_path):
        # Placeholder for Hume AI emotion analysis
        # In a real implementation, you would use the Hume API here
        await asyncio.sleep(1)  # Simulating API call
        return {
            "joy": 0.7,
            "neutral": 0.2,
            "excitement": 0.1
        }

    async def _generate_summary(self, transcript, emotion_data, timestamp):
        prompt = f"""
        {self.SYSTEM_PROMPT}

        Analyze the following transcript and emotion data for the timestamp {timestamp}:

        Transcript: {transcript}
        Emotion Data: {emotion_data}

        Provide a concise summary of the person's emotional state and content of speech.
        Focus on key points and any notable emotional shifts.

        Summary:
        """

        completion = self.groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )

        return completion.choices[0].message.content

# Usage example:
# async def main():
#     analyzer = SpeechToTextAnalyzer(groq_api_key="your_groq_key", hume_api_key="your_hume_key")
#     result = await analyzer.analyze_audio("path/to/audio.wav", timestamp=10.5)
#     print(result)
# 
# asyncio.run(main())