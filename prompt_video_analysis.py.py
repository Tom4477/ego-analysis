import google.generativeai as genai
import time

class VideoAnalyzer:
    SYSTEM_PROMPT = """
    You are an advanced AI assistant specializing in analyzing egocentric vision data to understand activities and environments within a luxurious mansion setting. Your task is to process video frames and provide insightful analysis of the person's surroundings and actions.

    Guidelines:
    1. Describe the environment in detail, focusing on the mansion's various areas and features.
    2. Identify the main activity of the person wearing the camera, considering typical activities in a mansion.
    3. Be specific and objective in your descriptions, noting luxurious or unique elements.
    4. Focus on observable details, not assumptions.
    5. Do not mention AI, image analysis, or the process of analysis.

    Your analysis should be clear, concise, and focused on the person's experience within the mansion setting.
    """

    ENVIRONMENTS = [
        "main hall", "living room", "dining room", "kitchen", "bedroom", "bathroom", 
        "study", "library", "home theater", "wine cellar", "indoor pool", "gym", 
        "garage", "garden", "patio", "balcony", "rooftop terrace", "front yard", 
        "backyard", "gazebo", "fountain area", "hedge maze", "tennis court", 
        "greenhouse", "guest house", "servant quarters", "unknown"
    ]

    ACTIVITIES = [
        "walking", "standing", "sitting", "lounging", "reading", "writing", 
        "using computer", "watching TV", "exercising", "swimming", "playing tennis", 
        "gardening", "cooking", "dining", "drinking tea/coffee", "having a cocktail", 
        "hosting guests", "cleaning", "organizing", "decorating", "admiring artwork", 
        "playing piano", "playing billiards", "meditating", "yoga", "unknown"
    ]

    def __init__(self, gemini_api_key):
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro')

    async def analyze_frame(self, frame_path, timestamp):
        prompt = f"""
        {self.SYSTEM_PROMPT}

        Analyze the following image frame from a first-person perspective video taken in a luxurious mansion at timestamp {timestamp}:

        1. Identify the specific area of the mansion. Choose the best fitting environment from this list: {', '.join(self.ENVIRONMENTS)}
        2. Describe the environment in detail, noting any luxurious features, decor, or unique elements visible.
        3. Identify the main activity of the person wearing the camera. Choose the best fitting activity from this list: {', '.join(self.ACTIVITIES)}
        4. Describe the activity in context of the mansion setting.
        5. Provide any additional relevant visual details that contribute to the sense of luxury or the mansion's atmosphere.

        If none of the listed environments or activities fit perfectly, choose the closest match and provide a brief explanation.

        Provide your response in this format:
        Environment: [chosen environment]
        Environment Description: [detailed description including luxurious features]
        Activity: [chosen activity]
        Activity Description: [detailed description in context of the mansion]
        Luxury Details: [additional relevant details emphasizing the mansion's luxury]

        Remember to be specific and focus on the unique aspects of the mansion setting.
        """

        response = await self.model.generate_content([frame_path, prompt])
        return self._parse_response(response.text, timestamp)

    def _parse_response(self, response_text, timestamp):
        lines = response_text.split('\n')
        environment = next((line.split(': ')[1] for line in lines if line.startswith('Environment: ')), 'unknown')
        activity = next((line.split(': ')[1] for line in lines if line.startswith('Activity: ')), 'unknown')
        env_description = next((line.split(': ')[1] for line in lines if line.startswith('Environment Description: ')), '')
        act_description = next((line.split(': ')[1] for line in lines if line.startswith('Activity Description: ')), '')
        luxury_details = next((line.split(': ')[1] for line in lines if line.startswith('Luxury Details: ')), '')

        return {
            "timestamp": timestamp,
            "environment": environment,
            "environment_description": env_description,
            "activity": activity,
            "activity_description": act_description,
            "luxury_details": luxury_details
        }

# Usage example:
# async def main():
#     analyzer = VideoAnalyzer(gemini_api_key="your_gemini_key")
#     result = await analyzer.analyze_frame("path/to/frame.jpg", timestamp=10.5)
#     print(result)
# 
# asyncio.run(main())