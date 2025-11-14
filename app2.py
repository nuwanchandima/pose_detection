from google import genai
from dotenv import load_dotenv
import os
import time
import json
from pydantic import BaseModel, Field
from typing import Any

load_dotenv(override=True)

def llm_call(video_path):

    try:

        client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        # video_path = "Rich Man Mocks Mom & Son  DramatizeMe - DramatizeMe (360p, h264).mp4"
        # video_path = "1112.mp4"
        # video_path = "1112(1).mp4"
        # video_path = "video.mp4"
        # video_path = "1112 (1).mp4"
        # video_path = "00-00-06-000_to_00-00-08-000__1.mp4"
        # video_path = "00-01-34-300_to_00-01-37-200__dialogue_11.mp4"
        # video_path = "00-01-46-000_to_00-01-47-000__11.mp4"

        class ExpectedOutputStructure(BaseModel):
            same_person: bool = Field(description="if same person then true else false")
            emotion: Any = Field(description="emotion of the person")
            reason: Any = Field(description="reason for the same person or different person or not found")

        # Upload the file
        myfile = client.files.upload(file=video_path)
        print(f"File uploaded: {myfile.name}")

        # Wait for the file to be processed
        print("Waiting for file to be processed...")
        while myfile.state.name == "PROCESSING":
            time.sleep(2)  # Wait 2 seconds before checking again
            myfile = client.files.get(name=myfile.name)
            print(f"Current state: {myfile.state.name}")

        # Check if the file is active and ready
        if myfile.state.name == "ACTIVE":
            print("File is ready! Generating content...")
            response = client.models.generate_content(
                # model="gemini-2.5-flash", 
                model="gemini-2.5-pro", 
                contents=[myfile, """
                tell me on this video talking person and visual person both are same or not . give me json like.i think you can do it with lip sync checking with voice

                    is different person then return,
                    {
                    "same_person":false (if different person)
                    "reason": "relevant reason",
                    "emotion": "null"
                    }
                    is same person then return,
                    {
                    "same_person":true (if same person),
                    "emotion": "relevant emotion"
                    "reason": "relevant reason"
                    }
                """],
                config={"temperature": 0.3, "response_schema": ExpectedOutputStructure},
            )
            print("Raw response:")
            print(response.text)
            print("\n" + "="*50 + "\n")
            
            # Try to extract JSON from the response
            response_text = response.text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith("```"):
                # Find the actual JSON content between code blocks
                lines = response_text.split("\n")
                json_lines = []
                in_code_block = False
                for line in lines:
                    if line.startswith("```"):
                        in_code_block = not in_code_block
                        continue
                    if in_code_block or (not line.startswith("```") and "{" in line):
                        json_lines.append(line)
                response_text = "\n".join(json_lines).strip()
            
            # If there's text before the JSON, try to extract just the JSON
            if not response_text.startswith("{"):
                # Find the first { and last }
                start = response_text.find("{")
                end = response_text.rfind("}")
                if start != -1 and end != -1:
                    response_text = response_text[start:end+1]
            
            print("Parsed JSON:")
            print(response_text)
            print("\n" + "="*50 + "\n")
            
            result = ExpectedOutputStructure.model_validate_json(response_text)
            print("Validated result:")
            print(result)

            # return parsed json as dictionary
            return json.loads(response_text)
        else:
            print(f"File processing failed. State: {myfile.state.name}")
            return {
                "error": True,
                "message": f"File processing failed. State: {myfile.state.name}"
            }
    except Exception as e:
        print(f"Error ------------> : {e}")
        return {
            "error": True,
            "message": str(e)
        }





# result = llm_call("00-01-34-300_to_00-01-37-200__dialogue_11.mp4")



# # Check for errors properly - result should always be a dict now
# if isinstance(result, dict) and result.get("error") == True:
#     # error happened
#     print(f"Error ------------> : {result}")

#     # expected output
#     # {'error': True, 'message': "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'The model is overloaded. Please try again later.', 'status': 'UNAVAILABLE'}}"}
# else:
#     # normal scenario
#     print("result------------->")
#     print(result)
    
#     # expected output
#     # {
#     # "same_person": true,
#     # "emotion": "Happy",
#     # "reason": "The man's lip movements at the beginning of the video are in sync with the spoken word 'Right'. He is also smiling, indicating a positive emotion."
#     # }