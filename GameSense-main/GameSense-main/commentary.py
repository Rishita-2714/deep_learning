try:
  import google.generativeai as genai
except Exception:
  genai = None
import pathlib
import textwrap
try:
  from IPython.display import display
  from IPython.display import Markdown
except Exception:
  display = None
  Markdown = None
import PIL.Image
import cv2
import os
import shutil
import os

model = None
model2 = None
# Prefer environment variable for the Gemini/Google API key to avoid committing secrets.
GEMINI_KEY = os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
if genai is not None and GEMINI_KEY:
  try:
    genai.configure(api_key=GEMINI_KEY)
    # Support both older client API (GenerativeModel) and newer function-based API
    if hasattr(genai, 'GenerativeModel'):
      model2 = genai.GenerativeModel('gemini-pro')
      model = genai.GenerativeModel('gemini-pro')
    elif hasattr(genai, 'generate_text') or hasattr(genai, 'generate'):
      # create simple callable wrappers that accept a prompt or image-path
      def _call_text_model(prompt_or_path, model_name='gemini-pro'):
        try:
          if hasattr(genai, 'generate_text'):
            resp = genai.generate_text(model=model_name, input=prompt_or_path)
            # try common response shapes
            if isinstance(resp, str):
              return resp
            if hasattr(resp, 'text'):
              return resp.text
            if isinstance(resp, dict):
              if 'candidates' in resp and resp['candidates']:
                cand = resp['candidates'][0]
                return cand.get('content') or cand.get('output') or str(cand)
              if 'output' in resp:
                return resp['output']
            return str(resp)
          elif hasattr(genai, 'generate'):
            resp = genai.generate(model=model_name, prompt=prompt_or_path)
            if isinstance(resp, dict) and 'candidates' in resp and resp['candidates']:
              return resp['candidates'][0].get('content') or str(resp['candidates'][0])
            return str(resp)
        except Exception:
          return None
      model2 = _call_text_model
      model = _call_text_model
    else:
      print('google.generativeai installed but no recognized API; disabling Gemini')
      model = None
      model2 = None
  except Exception as e:
    print(f"Error configuring Gemini API: {e}")
    model = None
    model2 = None
else:
  if genai is None:
    print("google.generativeai not installed; Gemini disabled")
  else:
    print("GEMINI_API_KEY not set; Gemini disabled")

# path_video = 'input_video/input_video_2.mp4'
# path_folder='output_frames'

#Generating Frames

def extract_frames(video_path, output_folder):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    frame_number = 0

    while success:
        # Save every frame as a .jpg file
        img_filename = os.path.join(output_folder, f"frame_{frame_number}.jpg")
        cv2.imwrite(img_filename, frame)
        # print(f"Saved: {img_filename}")

        # Read the next frame
        success, frame = video.read()
        frame_number += 1

    video.release()
    # print(f"Extraction complete. Saved {frame_number} frames in {output_folder}.")
    return frame_number

#Generating Comments

def generate_output(path_folder, path_video):
  commentary = []
  frame_num = extract_frames(path_video, path_folder)
  if model is None or model2 is None:
    print("Gemini API not configured, using default commentary")
    commentary = ["Commentary unavailable due to API error"] * 10  # default
  else:
    try:
      # response1 =""
      strtr = "" #Give the commentary of the following in a sentence in 10 words only:\n
      prompt ='you are a commentator for the game going on, please give the commentary on it and give more emphasis on the events going on such as type of shots being played, player actions, scoring etc in strictly atmost 5 words of one sentence only:\n'
      for i in range(0, frame_num, 20):
        path_image = os.path.join(path_folder, f'frame_{i}.jpg')
        try:
          # Try to get a short description from an image-aware model if available
          img_text = ""
          if hasattr(model2, 'generate_content'):
            with PIL.Image.open(path_image) as img:
              resp = model2.generate_content(img)
              img_text = getattr(resp, 'text', str(resp))
          elif callable(model2):
            # fallback to a text-only call that references the frame path
            img_text = model2(f"Describe the image at {path_image} in one short sentence.") or ""

          if i % 60 == 0:
            # finalize a chunk
            prompt_chunk = prompt + strtr + " " + img_text
            if hasattr(model, 'generate_content'):
              resp2 = model.generate_content(prompt_chunk)
              out = getattr(resp2, 'text', str(resp2))
            elif callable(model):
              out = model(prompt_chunk) or ""
            else:
              out = ""
            print(out)
            commentary.append(out)
            strtr = ""
          else:
            strtr = strtr + " " + img_text
        except Exception:
          continue
      if frame_num % 60 != 0 and strtr.strip():
          if hasattr(model, 'generate_content'):
            resp2 = model.generate_content(prompt + strtr)
            out = getattr(resp2, 'text', str(resp2))
          elif callable(model):
            out = model(prompt + strtr) or ""
          else:
            out = ""
          print(out)
          commentary.append(out)
    except Exception as e:
      print(f"Error in commentary generation: {e}")
      commentary = ["Commentary unavailable due to API error"] * 10  # default
  if os.path.exists(path_folder):
    shutil.rmtree(path_folder)
  return commentary, frame_num
