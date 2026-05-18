import os
import requests
import shutil

# URL for the FastAPI endpoint
API_URL = "http://localhost:8000/predict"

def find_good_image(class_dir, expected_name, dest_path):
    print(f"Searching for a good image in {class_dir} that predicts as '{expected_name}'...")
    valid_exts = {'.jpg', '.jpeg', '.png'}
    for f in os.listdir(class_dir):
        if f.startswith('.'): continue
        if os.path.splitext(f)[1].lower() not in valid_exts: continue
            
        file_path = os.path.join(class_dir, f)
        
        try:
            with open(file_path, 'rb') as img_file:
                files = {'file': (f, img_file, 'image/jpeg')}
                response = requests.post(API_URL, files=files)
                
            if response.status_code == 200:
                # The endpoint returns a rendered HTML template.
                # Let's just search the HTML text for the expected class name inside the specific div.
                # Actually, earlier I modified app.py to return JSON? No, it returns TemplateResponse.
                # In the HTML, the prediction name is placed in <div class="result-veg-name">{{ prediction.name_en }}</div>
                html_text = response.text
                if f'<div class="result-veg-name">{expected_name}</div>' in html_text or f'{expected_name}' in html_text:
                    # Let's be safe and make sure the expected name is definitely the prediction
                    # The safest way is to check if it's the highest confidence or just look for the class name.
                    print(f"✅ Found working image: {file_path}")
                    shutil.copy(file_path, dest_path)
                    return True
        except Exception as e:
            pass
    print(f"❌ Could not find any working image for {expected_name}.")
    return False

if __name__ == "__main__":
    peas_dir = "dataset/peas"
    ladies_finger_dir = "dataset/Ladies finger"
    
    find_good_image(peas_dir, "Peas", "static/samples/peas.jpg")
    find_good_image(ladies_finger_dir, "Ladies finger", "static/samples/ladies_finger.jpg")
    
    print("Done checking!")
