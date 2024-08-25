import os
import asyncio
from speech_to_text import SpeechToTextAnalyzer
from video_analysis import VideoAnalyzer

class FirstPersonVideoAnalyzer:
    OUTPUT_TEMPLATE = """
    At {timestamp} seconds into the video:
    
    Environment: The user is in the {environment} of the mansion.
    {environment_description}
    
    Activity: They are currently {activity}.
    {activity_description}
    
    Luxury Details: {luxury_details}
    
    Emotional State: The user appears to be feeling {emotions}.
    
    Summary: {summary}
    """

    def __init__(self, groq_api_key, hume_api_key, gemini_api_key):
        self.speech_analyzer = SpeechToTextAnalyzer(groq_api_key, hume_api_key)
        self.video_analyzer = VideoAnalyzer(gemini_api_key)

    async def analyze_video(self, video_path, duration):
        results = []
        for t in range(0, int(duration), 1):  # Analyze every second
            audio_path = self._extract_audio(video_path, t, t+10)
            frame_path = self._extract_frame(video_path, t)

            audio_result, video_result = await asyncio.gather(
                self.speech_analyzer.analyze_audio(audio_path, t),
                self.video_analyzer.analyze_frame(frame_path, t)
            )

            combined_result = self._combine_results(audio_result, video_result)
            results.append(combined_result)

            os.remove(audio_path)
            os.remove(frame_path)

        return results

    def _extract_audio(self, video_path, start_time, end_time):
        output_path = f"temp_audio_{start_time}.wav"
        os.system(f"ffmpeg -i {video_path} -ss {start_time} -to {end_time} -c copy {output_path}")
        return output_path

    def _extract_frame(self, video_path, timestamp):
        output_path = f"temp_frame_{timestamp}.jpg"
        os.system(f"ffmpeg -i {video_path} -ss {timestamp} -vframes 1 {output_path}")
        return output_path

    def _combine_results(self, audio_result, video_result):
        emotions = ', '.join(f"{emotion}: {score:.2f}" for emotion, score in audio_result['emotion'].items())
        return self.OUTPUT_TEMPLATE.format(
            timestamp=video_result['timestamp'],
            environment=video_result['environment'],
            environment_description=video_result['environment_description'],
            activity=video_result['activity'],
            activity_description=video_result['activity_description'],
            luxury_details=video_result['luxury_details'],
            emotions=emotions,
            summary=audio_result['summary']
        )

# Usage example:
# async def main():
#     analyzer = FirstPersonVideoAnalyzer(
#         groq_api_key="your_groq_key",
#         hume_api_key="your_hume_key",
#         gemini_api_key="your_gemini_key"
#     )
#     results = await analyzer.analyze_video("path/to/mansion_video.mp4", duration=60)
#     for result in results:
#         print(result)
#         print("---")
#
# asyncio.run(main())