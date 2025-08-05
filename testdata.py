# #######Original code #########
# # import cv2
# # import numpy as np
# # from keras.models import load_model
# # import random
# # import os

# # model=load_model('model_file.h5')

# # faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # labels_dict={0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}

# # # len(number_of_image), image_height, image_width, channel
# # ##Use for a single image 
# # image_path = input("Enter the full path to the image file: ").strip()

# # # Read the image
# # frame = cv2.imread(image_path)

# # # # Path to test directory
# # # test_dir = r"C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test"

# # # # Step 1: Get list of emotion folders
# # # emotion_folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]

# # # # Step 2: Choose a random folder
# # # chosen_folder = random.choice(emotion_folders)

# # # # Step 3: Get list of image files in that folder
# # # image_files = [f for f in os.listdir(chosen_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# # # # Step 4: Pick a random image
# # # chosen_image_path = os.path.join(chosen_folder, random.choice(image_files))

# # # print(f"Using image: {chosen_image_path}")

# # # # Load and process the image
# # # frame = cv2.imread(chosen_image_path)
# # if frame is None:
# #     print("Could not read the image. Please check the path.")
# # else:
# #     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# #     faces= faceDetect.detectMultiScale(gray, 1.3, 3)
# #     for x,y,w,h in faces:
# #         sub_face_img=gray[y:y+h, x:x+w]
# #         resized=cv2.resize(sub_face_img,(48,48))
# #         normalize=resized/255.0
# #         reshaped=np.reshape(normalize, (1, 48, 48, 1))
# #         result=model.predict(reshaped)
# #         label=np.argmax(result, axis=1)[0]
# #         print(label)
# #         cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
# #         cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
# #         cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
# #         cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
            
# #     cv2.imshow("Frame",frame)
# #     cv2.waitKey(0)
# #     cv2.destroyAllWindows()


# #####Modified code ######
# # import cv2
# # import numpy as np
# # from keras.models import load_model
# # import os
# # import random

# # # Load the trained model and face detector
# # model = load_model('model_file.h5')
# # faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # # Label dictionary
# # labels_dict = {
# #     0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
# #     4: 'Neutral', 5: 'Sad', 6: 'Surprise'
# # }

# # # Ask user for image path
# # image_path = input("Enter image path (or press Enter to use random test image): ").strip()

# # if not image_path:
# #     test_dir = r"C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test"
# #     folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
# #     random_folder = random.choice(folders)
# #     images = [f for f in os.listdir(random_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
# #     image_path = os.path.join(random_folder, random.choice(images))
# #     print(f"[INFO] Using random test image: {image_path}")

# # # Read image
# # frame = cv2.imread(image_path)

# # if frame is None:
# #     print("[ERROR] Could not load image. Check path.")
# #     exit()

# # # Resize for better viewing without distortion
# # scale_percent = 150  # 150% of original size
# # width = int(frame.shape[1] * scale_percent / 100)
# # height = int(frame.shape[0] * scale_percent / 100)
# # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

# # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# # print(f"[INFO] Faces detected: {len(faces)}")

# # if len(faces) == 0:
# #     print("[WARNING] No faces found!")
# # else:
# #     for x, y, w, h in faces:
# #     # Predict the emotion
# #         face_img = gray[y:y+h, x:x+w]
# #         resized = cv2.resize(face_img, (48, 48))
# #         normalized = resized / 255.0
# #         reshaped = np.reshape(normalized, (1, 48, 48, 1))
# #         result = model.predict(reshaped)
# #         label = labels_dict[np.argmax(result)]

# #         # Draw rectangle around the face
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

# #         # Draw label *above* the face rectangle
# #         label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
# #         label_y = max(y - 10, label_size[1] + 10)
# #         cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# #         print(f"[INFO] Detected emotion: {label}")

# # # Show result
# # cv2.imshow("Emotion Detection", frame)
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# ####Code with youtube shows any songs of its choice based on emotion ######
# # import cv2
# # import numpy as np
# # from keras.models import load_model
# # import os
# # import random
# # import webbrowser
# # import requests
# # from datetime import datetime, timedelta
# # import urllib.parse

# # # YouTube API Configuration
# # YOUTUBE_API_KEY = "your_youtube_api_key_here"
# # YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

# # # Emotion to search terms mapping (optimized for latest songs)
# # EMOTION_TO_SEARCH = {
# #     'Happy': ['happy songs ', 'upbeat music', 'feel good hits', 'pop songs'],
# #     'Sad': ['sad songs ', 'emotional music' , 'heartbreak songs', 'melancholy hits '],
# #     'Angry': ['rock songs ', 'metal hits', 'aggressive music', 'hard rock'],
# #     'Fear': ['dark music ', 'horror soundtrack', 'intense music', 'horror music '],
# #     'Surprise': ['trending songs ', 'viral hits', 'popular music', 'chart toppers '],
# #     'Disgust': ['alternative rock ', 'indie music', 'grunge songs', 'experimental music '],
# #     'Neutral': ['chill music ', 'relaxing songs', 'acoustic hits', 'indie songs ']
# # }

# # def search_youtube_videos(emotion, max_results=5):
# #     """Search for YouTube videos based on emotion using YouTube API"""
# #     if YOUTUBE_API_KEY == "your_youtube_api_key_here":
# #         print("[WARNING] No YouTube API key provided. Using web search fallback.")
# #         return search_youtube_web_fallback(emotion, max_results)
    
# #     search_terms = EMOTION_TO_SEARCH.get(emotion, EMOTION_TO_SEARCH['Neutral'])
# #     all_videos = []
    
# #     try:
# #         for search_term in search_terms[:2]:  # Use first 2 search terms
# #             # Calculate date for recent videos (last 6 months)
# #             six_months_ago = datetime.now() - timedelta(days=180)
# #             published_after = six_months_ago.strftime('%Y-%m-%dT%H:%M:%SZ')
            
# #             params = {
# #                 'part': 'snippet',
# #                 'q': search_term,
# #                 'type': 'video',
# #                 'maxResults': 3,
# #                 'order': 'relevance',
# #                 'publishedAfter': published_after,
# #                 'videoCategoryId': '10',  # Music category
# #                 'key': YOUTUBE_API_KEY
# #             }
            
# #             response = requests.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
            
# #             if response.status_code == 200:
# #                 data = response.json()
# #                 for item in data.get('items', []):
# #                     video_info = {
# #                         'title': item['snippet']['title'],
# #                         'channel': item['snippet']['channelTitle'],
# #                         'video_id': item['id']['videoId'],
# #                         'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
# #                         'thumbnail': item['snippet']['thumbnails']['medium']['url'],
# #                         'published': item['snippet']['publishedAt']
# #                     }
# #                     all_videos.append(video_info)
# #             else:
# #                 print(f"[WARNING] YouTube API request failed: {response.status_code}")
    
# #     except Exception as e:
# #         print(f"[ERROR] YouTube API search failed: {e}")
# #         return search_youtube_web_fallback(emotion, max_results)
    
# #     # Remove duplicates and return latest videos
# #     unique_videos = []
# #     seen_ids = set()
# #     for video in all_videos:
# #         if video['video_id'] not in seen_ids:
# #             unique_videos.append(video)
# #             seen_ids.add(video['video_id'])
    
# #     return unique_videos[:max_results]

# # def search_youtube_web_fallback(emotion, max_results=5):
# #     """Fallback method using direct YouTube search URLs (no API required)"""
# #     search_terms = EMOTION_TO_SEARCH.get(emotion, EMOTION_TO_SEARCH['Neutral'])
# #     videos = []
    
# #     # Create YouTube search URLs for each search term
# #     for i, search_term in enumerate(search_terms[:max_results]):
# #         encoded_term = urllib.parse.quote_plus(search_term)
# #         search_url = f"https://www.youtube.com/results?search_query={encoded_term}&sp=EgIQAQ%253D%253D"  # Filter for videos
        
# #         video_info = {
# #             'title': f"{emotion} Music - {search_term.title()}",
# #             'channel': 'YouTube Search',
# #             'video_id': f'search_{i}',
# #             'url': search_url,
# #             'thumbnail': '',
# #             'published': 'Recent'
# #         }
# #         videos.append(video_info)
    
# #     return videos

# # def get_trending_music():
# #     """Get trending music videos"""
# #     if YOUTUBE_API_KEY == "your_youtube_api_key_here":
# #         return []
    
# #     try:
# #         params = {
# #             'part': 'snippet',
# #             'chart': 'mostPopular',
# #             'regionCode': 'US',
# #             'videoCategoryId': '10',  # Music category
# #             'maxResults': 10,
# #             'key': YOUTUBE_API_KEY
# #         }
        
# #         response = requests.get(f"{YOUTUBE_API_BASE_URL}/videos", params=params)
        
# #         if response.status_code == 200:
# #             data = response.json()
# #             videos = []
# #             for item in data.get('items', []):
# #                 video_info = {
# #                     'title': item['snippet']['title'],
# #                     'channel': item['snippet']['channelTitle'],
# #                     'video_id': item['id'],
# #                     'url': f"https://www.youtube.com/watch?v={item['id']}",
# #                     'thumbnail': item['snippet']['thumbnails']['medium']['url'],
# #                     'published': item['snippet']['publishedAt']
# #                 }
# #                 videos.append(video_info)
# #             return videos
# #     except Exception as e:
# #         print(f"[ERROR] Failed to get trending music: {e}")
    
# #     return []

# # def display_video_recommendations(videos, emotion):
# #     """Display recommended videos"""
# #     print(f"\n[INFO] YouTube music recommendations for {emotion} emotion:")
# #     print("=" * 70)
    
# #     for i, video in enumerate(videos, 1):
# #         print(f"{i}. {video['title']}")
# #         print(f"   Channel: {video['channel']}")
# #         print(f"   URL: {video['url']}")
# #         if video.get('published'):
# #             print(f"   Published: {video['published']}")
# #         print()

# # def open_videos_in_browser(videos, open_all=False):
# #     """Open videos in browser"""
# #     if not videos:
# #         print("[WARNING] No videos to open.")
# #         return
    
# #     if open_all:
# #         print(f"[INFO] Opening {len(videos)} videos in browser...")
# #         for video in videos:
# #             webbrowser.open(video['url'])
# #             print(f"   Opened: {video['title']}")
# #     else:
# #         print(f"[INFO] Opening first video in browser...")
# #         webbrowser.open(videos[0]['url'])
# #         print(f"   Opened: {videos[0]['title']}")

# # def create_youtube_playlist_url(videos, emotion):
# #     """Create a YouTube playlist URL (requires manual creation)"""
# #     if not videos:
# #         return None
    
# #     # Create a search URL that includes multiple songs
# #     search_terms = " OR ".join([f'"{video["title"]}"' for video in videos[:3]])
# #     encoded_search = urllib.parse.quote_plus(search_terms)
# #     playlist_search_url = f"https://www.youtube.com/results?search_query={encoded_search}"
    
# #     return playlist_search_url

# # # Load the trained model and face detector
# # model = load_model('model_file.h5')
# # faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # # Label dictionary
# # labels_dict = {
# #     0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
# #     4: 'Neutral', 5: 'Sad', 6: 'Surprise'
# # }

# # print("[INFO] Emotion Detection with YouTube Music Integration")
# # print("=" * 50)

# # # Optional: Set up YouTube API key
# # api_setup = input("Do you have a YouTube API key? (y/n - press n for basic functionality): ").lower().strip()
# # if api_setup == 'y':
# #     api_key = input("Enter your YouTube API key: ").strip()
# #     if api_key:
# #         YOUTUBE_API_KEY = api_key
# #         print("[INFO] YouTube API configured!")
# #     else:
# #         print("[INFO] Using web search fallback mode.")
# # else:
# #     print("[INFO] Using web search fallback mode (no API required).")

# # # Ask user for image path
# # image_path = input("\nEnter image path (or press Enter to use random test image): ").strip()

# # if not image_path:
# #     test_dir = r"C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test"
# #     if os.path.exists(test_dir):
# #         folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
# #         if folders:
# #             random_folder = random.choice(folders)
# #             images = [f for f in os.listdir(random_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
# #             if images:
# #                 image_path = os.path.join(random_folder, random.choice(images))
# #                 print(f"[INFO] Using random test image: {image_path}")
# #             else:
# #                 print("[ERROR] No images found in test directory.")
# #                 exit()
# #         else:
# #             print("[ERROR] No emotion folders found in test directory.")
# #             exit()
# #     else:
# #         image_path = input("Test directory not found. Please enter full image path: ").strip()

# # # Read image
# # frame = cv2.imread(image_path)

# # if frame is None:
# #     print("[ERROR] Could not load image. Check path.")
# #     exit()

# # # Resize for better viewing without distortion
# # scale_percent = 150  # 150% of original size
# # width = int(frame.shape[1] * scale_percent / 100)
# # height = int(frame.shape[0] * scale_percent / 100)
# # frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

# # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# # print(f"\n[INFO] Faces detected: {len(faces)}")

# # detected_emotions = []

# # if len(faces) == 0:
# #     print("[WARNING] No faces found!")
# # else:
# #     for x, y, w, h in faces:
# #         # Predict the emotion
# #         face_img = gray[y:y+h, x:x+w]
# #         resized = cv2.resize(face_img, (48, 48))
# #         normalized = resized / 255.0
# #         reshaped = np.reshape(normalized, (1, 48, 48, 1))
# #         result = model.predict(reshaped)
# #         label = labels_dict[np.argmax(result)]
        
# #         detected_emotions.append(label)
        
# #         # Draw rectangle around the face
# #         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
# #         # Draw label *above* the face rectangle
# #         label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
# #         label_y = max(y - 10, label_size[1] + 10)
# #         cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
# #         print(f"[INFO] Detected emotion: {label}")

# # # Show result
# # cv2.imshow("Emotion Detection", frame)

# # # Process YouTube integration if emotion detected
# # if detected_emotions:
# #     # Use the most common emotion if multiple faces
# #     primary_emotion = max(set(detected_emotions), key=detected_emotions.count)
    
# #     print(f"\n[INFO] Primary emotion detected: {primary_emotion}")
# #     print("[INFO] Searching for latest YouTube music...")
    
# #     # Search for videos
# #     recommended_videos = search_youtube_videos(primary_emotion)
    
# #     if recommended_videos:
# #         # Display recommendations
# #         display_video_recommendations(recommended_videos, primary_emotion)
        
# #         # Ask user what they want to do
# #         print("Choose an option:")
# #         print("1. Open first video")
# #         print("2. Open all videos")
# #         print("3. Just show recommendations (no opening)")
        
# #         choice = input("Enter choice (1-3): ").strip()
        
# #         if choice == '1':
# #             open_videos_in_browser(recommended_videos, open_all=False)
# #         elif choice == '2':
# #             open_videos_in_browser(recommended_videos, open_all=True)
# #         elif choice == '3':
# #             print("[INFO] Recommendations displayed. You can manually click the URLs above.")
# #         else:
# #             print("[INFO] Invalid choice. Opening first video...")
# #             open_videos_in_browser(recommended_videos, open_all=False)
# #     else:
# #         print(f"[WARNING] No videos found for {primary_emotion} emotion.")
# #         # Fallback: open generic YouTube music search
# #         fallback_search = urllib.parse.quote_plus(f"{primary_emotion} music 2024")
# #         fallback_url = f"https://www.youtube.com/results?search_query={fallback_search}"
# #         print(f"[INFO] Opening generic search: {fallback_url}")
# #         webbrowser.open(fallback_url)

# # else:
# #     print("\n[INFO] No emotions detected. Cannot recommend music.")

# # print("\nPress any key to close the image window...")
# # cv2.waitKey(0)
# # cv2.destroyAllWindows()
# # print("\n[INFO] Program completed successfully!")


# ##### code with choice of specific single song or playlist based on emotion ######
# import cv2
# import numpy as np
# from keras.models import load_model
# import os
# import random
# import webbrowser
# import requests
# from datetime import datetime, timedelta
# import urllib.parse
# import json

# # YouTube API Configuration
# YOUTUBE_API_KEY = "your_youtube_api_key_here"
# YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

# # Emotion to search terms mapping for artists and playlists
# EMOTION_TO_ARTISTS = {
#     'Happy': ['Ed Sheeran', 'Bruno Mars', 'Dua Lipa', 'The Weeknd', 'Taylor Swift', 'Post Malone'],
#     'Sad': ['Billie Eilish', 'Lana Del Rey', 'Adele', 'Sam Smith', 'Lewis Capaldi', 'Olivia Rodrigo'],
#     'Angry': ['Eminem', 'Linkin Park', 'Imagine Dragons', 'Twenty One Pilots', 'Metallica', 'Green Day'],
#     'Fear': ['Hans Zimmer', 'Trent Reznor', 'John Carpenter', 'Akira Yamaoka', 'Silent Hill OST', 'Horror Music'],
#     'Surprise': ['Doja Cat', 'Lil Nas X', 'Bad Bunny', 'BTS', 'Blackpink', 'Trending Artists'],
#     'Disgust': ['Arctic Monkeys', 'Radiohead', 'Nirvana', 'Pearl Jam', 'Foo Fighters', 'Red Hot Chili Peppers'],
#     'Neutral': ['Lofi Girl', 'Bon Iver', 'Phoebe Bridgers', 'Mac DeMarco', 'Tame Impala', 'Indie Folk']
# }

# EMOTION_TO_PLAYLIST_TERMS = {
#     'Happy': ['happy playlist 2024', 'feel good music playlist', 'upbeat songs playlist', 'pop hits 2024'],
#     'Sad': ['sad songs playlist 2024', 'heartbreak playlist', 'emotional music playlist', 'crying songs'],
#     'Angry': ['rock playlist 2024', 'metal music playlist', 'aggressive songs playlist', 'workout music'],
#     'Fear': ['dark music playlist', 'horror soundtrack playlist', 'scary music playlist', 'thriller music'],
#     'Surprise': ['trending playlist 2024', 'viral songs playlist', 'top hits 2024', 'popular music playlist'],
#     'Disgust': ['alternative rock playlist', 'indie playlist 2024', 'grunge music playlist', 'underground music'],
#     'Neutral': ['chill playlist 2024', 'relaxing music playlist', 'study music playlist', 'ambient playlist']
# }

# def search_artist_songs(artist_name, emotion, max_results=10):
#     """Search for songs by a specific artist with 3-5 minute duration"""
#     if YOUTUBE_API_KEY == "your_youtube_api_key_here":
#         return search_artist_web_fallback(artist_name, emotion, max_results)
    
#     try:
#         # Search for artist's songs
#         search_query = f"{artist_name} songs"
        
#         params = {
#             'part': 'snippet,contentDetails',
#             'q': search_query,
#             'type': 'video',
#             'maxResults': max_results * 2,  # Get more to filter by duration
#             'order': 'relevance',
#             'videoCategoryId': '10',  # Music category
#             'key': YOUTUBE_API_KEY
#         }
        
#         response = requests.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
        
#         if response.status_code == 200:
#             data = response.json()
#             video_ids = [item['id']['videoId'] for item in data.get('items', [])]
            
#             # Get video details including duration
#             if video_ids:
#                 details_params = {
#                     'part': 'snippet,contentDetails',
#                     'id': ','.join(video_ids),
#                     'key': YOUTUBE_API_KEY
#                 }
                
#                 details_response = requests.get(f"{YOUTUBE_API_BASE_URL}/videos", params=details_params)
                
#                 if details_response.status_code == 200:
#                     details_data = details_response.json()
#                     songs = []
                    
#                     for item in details_data.get('items', []):
#                         duration = item['contentDetails']['duration']
#                         duration_seconds = parse_duration(duration)
                        
#                         # Filter for songs between 2-6 minutes (120-360 seconds)
#                         if 120 <= duration_seconds <= 360:
#                             song_info = {
#                                 'title': item['snippet']['title'],
#                                 'channel': item['snippet']['channelTitle'],
#                                 'video_id': item['id'],
#                                 'url': f"https://www.youtube.com/watch?v={item['id']}",
#                                 'duration': format_duration(duration_seconds),
#                                 'thumbnail': item['snippet']['thumbnails']['medium']['url'],
#                                 'published': item['snippet']['publishedAt']
#                             }
#                             # Only include if it's likely the actual artist
#                             if artist_name.lower() in item['snippet']['channelTitle'].lower() or \
#                                artist_name.lower() in item['snippet']['title'].lower():
#                                 songs.append(song_info)
                    
#                     return songs[:max_results]
    
#     except Exception as e:
#         print(f"[ERROR] Artist search failed: {e}")
#         return search_artist_web_fallback(artist_name, emotion, max_results)
    
#     return []

# def search_artist_web_fallback(artist_name, emotion, max_results=10):
#     """Fallback method for artist search"""
#     search_term = f"{artist_name} songs 3 minutes"
#     encoded_term = urllib.parse.quote_plus(search_term)
#     search_url = f"https://www.youtube.com/results?search_query={encoded_term}&sp=EgIQAQ%253D%253D"
    
#     return [{
#         'title': f"{artist_name} - Songs Collection",
#         'channel': artist_name,
#         'video_id': 'search_artist',
#         'url': search_url,
#         'duration': '~3 min each',
#         'thumbnail': '',
#         'published': 'Various'
#     }]

# def search_playlists(emotion, max_results=8):
#     """Search for playlists based on emotion"""
#     if YOUTUBE_API_KEY == "your_youtube_api_key_here":
#         return search_playlists_web_fallback(emotion, max_results)
    
#     playlist_terms = EMOTION_TO_PLAYLIST_TERMS.get(emotion, EMOTION_TO_PLAYLIST_TERMS['Neutral'])
#     all_playlists = []
    
#     try:
#         for search_term in playlist_terms[:3]:  # Use first 3 terms
#             params = {
#                 'part': 'snippet',
#                 'q': search_term,
#                 'type': 'playlist',
#                 'maxResults': 3,
#                 'order': 'relevance',
#                 'key': YOUTUBE_API_KEY
#             }
            
#             response = requests.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
            
#             if response.status_code == 200:
#                 data = response.json()
#                 for item in data.get('items', []):
#                     playlist_info = {
#                         'title': item['snippet']['title'],
#                         'channel': item['snippet']['channelTitle'],
#                         'playlist_id': item['id']['playlistId'],
#                         'url': f"https://www.youtube.com/playlist?list={item['id']['playlistId']}",
#                         'thumbnail': item['snippet']['thumbnails']['medium']['url'],
#                         'published': item['snippet']['publishedAt']
#                     }
#                     all_playlists.append(playlist_info)
    
#     except Exception as e:
#         print(f"[ERROR] Playlist search failed: {e}")
#         return search_playlists_web_fallback(emotion, max_results)
    
#     # Remove duplicates
#     unique_playlists = []
#     seen_ids = set()
#     for playlist in all_playlists:
#         if playlist['playlist_id'] not in seen_ids:
#             unique_playlists.append(playlist)
#             seen_ids.add(playlist['playlist_id'])
    
#     return unique_playlists[:max_results]

# def search_playlists_web_fallback(emotion, max_results=8):
#     """Fallback method for playlist search"""
#     playlist_terms = EMOTION_TO_PLAYLIST_TERMS.get(emotion, EMOTION_TO_PLAYLIST_TERMS['Neutral'])
#     playlists = []
    
#     for i, search_term in enumerate(playlist_terms[:max_results]):
#         encoded_term = urllib.parse.quote_plus(search_term)
#         search_url = f"https://www.youtube.com/results?search_query={encoded_term}&sp=EgIQAw%253D%253D"  # Filter for playlists
        
#         playlist_info = {
#             'title': search_term.title(),
#             'channel': 'Various Artists',
#             'playlist_id': f'search_{i}',
#             'url': search_url,
#             'thumbnail': '',
#             'published': 'Various'
#         }
#         playlists.append(playlist_info)
    
#     return playlists

# def parse_duration(duration_str):
#     """Parse YouTube duration format (PT4M13S) to seconds"""
#     import re
#     pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
#     match = re.match(pattern, duration_str)
    
#     if not match:
#         return 0
    
#     hours = int(match.group(1) or 0)
#     minutes = int(match.group(2) or 0)
#     seconds = int(match.group(3) or 0)
    
#     return hours * 3600 + minutes * 60 + seconds

# def format_duration(seconds):
#     """Format seconds to MM:SS"""
#     minutes = seconds // 60
#     seconds = seconds % 60
#     return f"{minutes}:{seconds:02d}"

# def display_artists_menu(artists, emotion):
#     """Display artists menu for selection"""
#     print(f"\n[INFO] Popular {emotion} Artists:")
#     print("=" * 50)
    
#     for i, artist in enumerate(artists, 1):
#         print(f"{i}. {artist}")
    
#     print(f"{len(artists) + 1}. Search for a custom artist")
#     print("0. Go back to main menu")
    
#     return artists

# def display_artist_songs(songs, artist_name):
#     """Display songs by selected artist"""
#     print(f"\n[INFO] Songs by {artist_name} (3-5 minute range):")
#     print("=" * 60)
    
#     if not songs:
#         print("No songs found or using web search mode.")
#         return
    
#     for i, song in enumerate(songs, 1):
#         print(f"{i}. {song['title']}")
#         print(f"   Channel: {song['channel']}")
#         print(f"   Duration: {song['duration']}")
#         print(f"   URL: {song['url']}")
#         print()

# def display_playlists_menu(playlists, emotion):
#     """Display playlists menu for selection"""
#     print(f"\n[INFO] {emotion} Music Playlists:")
#     print("=" * 50)
    
#     for i, playlist in enumerate(playlists, 1):
#         print(f"{i}. {playlist['title']}")
#         print(f"   Channel: {playlist['channel']}")
#         print(f"   URL: {playlist['url']}")
#         print()
    
#     print("0. Go back to main menu")

# def get_user_choice(max_choice, prompt="Enter your choice: "):
#     """Get user choice with validation"""
#     while True:
#         try:
#             choice = int(input(prompt))
#             if 0 <= choice <= max_choice:
#                 return choice
#             else:
#                 print(f"Please enter a number between 0 and {max_choice}")
#         except ValueError:
#             print("Please enter a valid number")

# # Load the trained model and face detector
# model = load_model('model_file.h5')
# faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Label dictionary
# labels_dict = {
#     0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
#     4: 'Neutral', 5: 'Sad', 6: 'Surprise'
# }

# print("[INFO] Emotion Detection with Enhanced YouTube Music Integration")
# print("=" * 60)

# # Optional: Set up YouTube API key
# api_setup = input("Do you have a YouTube API key for better results? (y/n): ").lower().strip()
# if api_setup == 'y':
#     api_key = input("Enter your YouTube API key: ").strip()
#     if api_key:
#         YOUTUBE_API_KEY = api_key
#         print("[INFO] YouTube API configured for enhanced features!")
#     else:
#         print("[INFO] Using web search fallback mode.")
# else:
#     print("[INFO] Using web search fallback mode (basic functionality).")

# # Ask user for image path
# image_path = input("\nEnter image path (or press Enter to use random test image): ").strip()

# if not image_path:
#     test_dir = r"C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test"
#     if os.path.exists(test_dir):
#         folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
#         if folders:
#             random_folder = random.choice(folders)
#             images = [f for f in os.listdir(random_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#             if images:
#                 image_path = os.path.join(random_folder, random.choice(images))
#                 print(f"[INFO] Using random test image: {image_path}")
#             else:
#                 print("[ERROR] No images found in test directory.")
#                 exit()
#         else:
#             print("[ERROR] No emotion folders found in test directory.")
#             exit()
#     else:
#         image_path = input("Test directory not found. Please enter full image path: ").strip()

# # Read image
# frame = cv2.imread(image_path)

# if frame is None:
#     print("[ERROR] Could not load image. Check path.")
#     exit()

# # Resize for better viewing without distortion
# scale_percent = 150  # 150% of original size
# width = int(frame.shape[1] * scale_percent / 100)
# height = int(frame.shape[0] * scale_percent / 100)
# frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# print(f"\n[INFO] Faces detected: {len(faces)}")

# detected_emotions = []

# if len(faces) == 0:
#     print("[WARNING] No faces found!")
# else:
#     for x, y, w, h in faces:
#         # Predict the emotion
#         face_img = gray[y:y+h, x:x+w]
#         resized = cv2.resize(face_img, (48, 48))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 48, 48, 1))
#         result = model.predict(reshaped)
#         label = labels_dict[np.argmax(result)]
        
#         detected_emotions.append(label)
        
#         # Draw rectangle around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
#         # Draw label *above* the face rectangle
#         label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#         label_y = max(y - 10, label_size[1] + 10)
#         cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         print(f"[INFO] Detected emotion: {label}")

# # Show result
# cv2.imshow("Emotion Detection", frame)

# # Process YouTube integration if emotion detected
# if detected_emotions:
#     # Use the most common emotion if multiple faces
#     primary_emotion = max(set(detected_emotions), key=detected_emotions.count)
    
#     print(f"\n[INFO] Primary emotion detected: {primary_emotion}")
    
#     while True:
#         print(f"\n[MENU] What would you like to listen to for your {primary_emotion} mood?")
#         print("=" * 60)
#         print("1. Single Artist Songs (3-5 minute songs)")
#         print("2. Music Playlists")
#         print("3. Exit")
        
#         main_choice = get_user_choice(3, "Choose option (1-3): ")
        
#         if main_choice == 1:
#             # Single Artist Mode
#             artists = EMOTION_TO_ARTISTS.get(primary_emotion, EMOTION_TO_ARTISTS['Neutral'])
#             display_artists_menu(artists, primary_emotion)
            
#             artist_choice = get_user_choice(len(artists) + 1, f"Choose artist (0-{len(artists) + 1}): ")
            
#             if artist_choice == 0:
#                 continue  # Go back to main menu
#             elif artist_choice == len(artists) + 1:
#                 # Custom artist search
#                 custom_artist = input("Enter artist name: ").strip()
#                 if custom_artist:
#                     selected_artist = custom_artist
#                 else:
#                     continue
#             else:
#                 selected_artist = artists[artist_choice - 1]
            
#             print(f"\n[INFO] Searching for songs by {selected_artist}...")
#             artist_songs = search_artist_songs(selected_artist, primary_emotion)
            
#             display_artist_songs(artist_songs, selected_artist)
            
#             if artist_songs:
#                 open_choice = input(f"\nOpen {selected_artist}'s music page? (y/n): ").lower().strip()
#                 if open_choice == 'y':
#                     if len(artist_songs) == 1 and artist_songs[0]['video_id'] == 'search_artist':
#                         # Web fallback mode
#                         webbrowser.open(artist_songs[0]['url'])
#                         print(f"[INFO] Opened YouTube search for {selected_artist}")
#                     else:
#                         # Open artist's channel or first song
#                         webbrowser.open(artist_songs[0]['url'])
#                         print(f"[INFO] Opened first song by {selected_artist}")
        
#         elif main_choice == 2:
#             # Playlist Mode
#             print(f"\n[INFO] Searching for {primary_emotion} playlists...")
#             playlists = search_playlists(primary_emotion)
            
#             display_playlists_menu(playlists, primary_emotion)
            
#             playlist_choice = get_user_choice(len(playlists), f"Choose playlist (0-{len(playlists)}): ")
            
#             if playlist_choice == 0:
#                 continue  # Go back to main menu
            
#             selected_playlist = playlists[playlist_choice - 1]
            
#             open_choice = input(f"\nOpen '{selected_playlist['title']}'? (y/n): ").lower().strip()
#             if open_choice == 'y':
#                 webbrowser.open(selected_playlist['url'])
#                 print(f"[INFO] Opened playlist: {selected_playlist['title']}")
        
#         elif main_choice == 3:
#             print("[INFO] Exiting music selection...")
#             break

# else:
#     print("\n[INFO] No emotions detected. Cannot recommend music.")

# print("\nPress any key to close the image window...")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("\n[INFO] Program completed successfully!")






#####Code with better search for artist and playlist, and option to open specific song or playlist######    






# import cv2
# import numpy as np
# from keras.models import load_model
# import os
# import random
# import webbrowser
# import requests
# from datetime import datetime, timedelta
# import urllib.parse
# import json

# # YouTube API Configuration
# YOUTUBE_API_KEY = "your_youtube_api_key_here"
# YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

# # Emotion to search terms mapping for artists and playlists
# EMOTION_TO_ARTISTS = {
#     'Happy': ['Ed Sheeran', 'Bruno Mars', 'Dua Lipa', 'The Weeknd', 'Taylor Swift', 'Post Malone'],
#     'Sad': ['Billie Eilish', 'Lana Del Rey', 'Adele', 'Sam Smith', 'Lewis Capaldi', 'Olivia Rodrigo'],
#     'Angry': ['Eminem', 'Linkin Park', 'Imagine Dragons', 'Twenty One Pilots', 'Metallica', 'Green Day'],
#     'Fear': ['Hans Zimmer', 'Trent Reznor', 'John Carpenter', 'Akira Yamaoka', 'Silent Hill OST', 'Horror Music'],
#     'Surprise': ['Doja Cat', 'Lil Nas X', 'Bad Bunny', 'BTS', 'Blackpink', 'Trending Artists'],
#     'Disgust': ['Arctic Monkeys', 'Radiohead', 'Nirvana', 'Pearl Jam', 'Foo Fighters', 'Red Hot Chili Peppers'],
#     'Neutral': ['Lofi Girl', 'Bon Iver', 'Phoebe Bridgers', 'Mac DeMarco', 'Tame Impala', 'Indie Folk']
# }

# EMOTION_TO_PLAYLIST_TERMS = {
#     'Happy': ['happy playlist 2024', 'feel good music playlist', 'upbeat songs playlist', 'pop hits 2024'],
#     'Sad': ['sad songs playlist 2024', 'heartbreak playlist', 'emotional music playlist', 'crying songs'],
#     'Angry': ['rock playlist 2024', 'metal music playlist', 'aggressive songs playlist', 'workout music'],
#     'Fear': ['dark music playlist', 'horror soundtrack playlist', 'scary music playlist', 'thriller music'],
#     'Surprise': ['trending playlist 2024', 'viral songs playlist', 'top hits 2024', 'popular music playlist'],
#     'Disgust': ['alternative rock playlist', 'indie playlist 2024', 'grunge music playlist', 'underground music'],
#     'Neutral': ['chill playlist 2024', 'relaxing music playlist', 'study music playlist', 'ambient playlist']
# }

# def search_artist_songs(artist_name, emotion, max_results=10):
#     """Search for songs by a specific artist with 3-5 minute duration"""
#     if YOUTUBE_API_KEY == "your_youtube_api_key_here":
#         return search_artist_web_fallback(artist_name, emotion, max_results)
    
#     try:
#         # Try multiple search strategies
#         search_queries = [
#             f"{artist_name} official songs",
#             f"{artist_name} music videos",
#             f"{artist_name} hits",
#             f"{artist_name} best songs"
#         ]
        
#         all_songs = []
        
#         for search_query in search_queries[:2]:  # Try first 2 queries
#             print(f"[DEBUG] Searching with query: {search_query}")
            
#             params = {
#                 'part': 'snippet',
#                 'q': search_query,
#                 'type': 'video',
#                 'maxResults': 15,  # Get more results
#                 'order': 'relevance',
#                 'videoCategoryId': '10',  # Music category
#                 'key': YOUTUBE_API_KEY
#             }
            
#             response = requests.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
#             print(f"[DEBUG] Search response status: {response.status_code}")
            
#             if response.status_code == 200:
#                 data = response.json()
#                 items = data.get('items', [])
#                 print(f"[DEBUG] Found {len(items)} videos")
                
#                 if items:
#                     video_ids = [item['id']['videoId'] for item in items]
                    
#                     # Get video details including duration
#                     details_params = {
#                         'part': 'snippet,contentDetails',
#                         'id': ','.join(video_ids),
#                         'key': YOUTUBE_API_KEY
#                     }
                    
#                     details_response = requests.get(f"{YOUTUBE_API_BASE_URL}/videos", params=details_params)
#                     print(f"[DEBUG] Details response status: {details_response.status_code}")
                    
#                     if details_response.status_code == 200:
#                         details_data = details_response.json()
#                         detail_items = details_data.get('items', [])
#                         print(f"[DEBUG] Got details for {len(detail_items)} videos")
                        
#                         for item in detail_items:
#                             try:
#                                 duration = item['contentDetails']['duration']
#                                 duration_seconds = parse_duration(duration)
                                
#                                 # More lenient duration filter: 1-8 minutes (60-480 seconds)
#                                 if 60 <= duration_seconds <= 480:
#                                     title = item['snippet']['title']
#                                     channel = item['snippet']['channelTitle']
                                    
#                                     # More flexible artist matching
#                                     artist_in_title = any(word in title.lower() for word in artist_name.lower().split())
#                                     artist_in_channel = any(word in channel.lower() for word in artist_name.lower().split())
                                    
#                                     # Include if artist name appears in title or channel
#                                     if artist_in_title or artist_in_channel:
#                                         song_info = {
#                                             'title': title,
#                                             'channel': channel,
#                                             'video_id': item['id'],
#                                             'url': f"https://www.youtube.com/watch?v={item['id']}",
#                                             'duration': format_duration(duration_seconds),
#                                             'thumbnail': item['snippet']['thumbnails']['medium']['url'],
#                                             'published': item['snippet']['publishedAt'][:10]  # Just date
#                                         }
#                                         all_songs.append(song_info)
#                                         print(f"[DEBUG] Added song: {title[:50]}...")
#                             except Exception as e:
#                                 print(f"[DEBUG] Error processing video: {e}")
#                                 continue
#             else:
#                 print(f"[ERROR] Search API error: {response.status_code}")
#                 if response.status_code == 403:
#                     print("[ERROR] API quota exceeded or invalid key")
#                     return search_artist_web_fallback(artist_name, emotion, max_results)
        
#         # Remove duplicates based on video_id
#         unique_songs = []
#         seen_ids = set()
#         for song in all_songs:
#             if song['video_id'] not in seen_ids:
#                 unique_songs.append(song)
#                 seen_ids.add(song['video_id'])
        
#         print(f"[DEBUG] Total unique songs found: {len(unique_songs)}")
        
#         if unique_songs:
#             return unique_songs[:max_results]
#         else:
#             print("[DEBUG] No songs found with API, using web fallback")
#             return search_artist_web_fallback(artist_name, emotion, max_results)
    
#     except Exception as e:
#         print(f"[ERROR] Artist search failed: {e}")
#         return search_artist_web_fallback(artist_name, emotion, max_results)

# def search_artist_web_fallback(artist_name, emotion, max_results=10):
#     """Fallback method for artist search - creates multiple search options"""
#     print(f"[INFO] Using web search for {artist_name}")
    
#     search_options = [
#         f"{artist_name} songs",
#         f"{artist_name} hits",
#         f"{artist_name} best songs",
#         f"{artist_name} official music videos",
#         f"{artist_name} top tracks"
#     ]
    
#     fallback_results = []
    
#     for i, search_term in enumerate(search_options[:max_results]):
#         encoded_term = urllib.parse.quote_plus(search_term)
#         search_url = f"https://www.youtube.com/results?search_query={encoded_term}&sp=EgIQAQ%253D%253D"
        
#         song_info = {
#             'title': f"{search_term.title()}",
#             'channel': 'YouTube Search',
#             'video_id': f'search_{i}',
#             'url': search_url,
#             'duration': 'Various',
#             'thumbnail': '',
#             'published': 'Recent'
#         }
#         fallback_results.append(song_info)
    
#     return fallback_results










# def search_playlists(emotion, max_results=8):
#     """Search for playlists based on emotion"""
#     if YOUTUBE_API_KEY == "your_youtube_api_key_here":
#         return search_playlists_web_fallback(emotion, max_results)
    
#     playlist_terms = EMOTION_TO_PLAYLIST_TERMS.get(emotion, EMOTION_TO_PLAYLIST_TERMS['Neutral'])
#     all_playlists = []
    
#     try:
#         for search_term in playlist_terms[:3]:  # Use first 3 terms
#             params = {
#                 'part': 'snippet',
#                 'q': search_term,
#                 'type': 'playlist',
#                 'maxResults': 3,
#                 'order': 'relevance',
#                 'key': YOUTUBE_API_KEY
#             }
            
#             response = requests.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
            
#             if response.status_code == 200:
#                 data = response.json()
#                 for item in data.get('items', []):
#                     playlist_info = {
#                         'title': item['snippet']['title'],
#                         'channel': item['snippet']['channelTitle'],
#                         'playlist_id': item['id']['playlistId'],
#                         'url': f"https://www.youtube.com/playlist?list={item['id']['playlistId']}",
#                         'thumbnail': item['snippet']['thumbnails']['medium']['url'],
#                         'published': item['snippet']['publishedAt']
#                     }
#                     all_playlists.append(playlist_info)
    
#     except Exception as e:
#         print(f"[ERROR] Playlist search failed: {e}")
#         return search_playlists_web_fallback(emotion, max_results)
    
#     # Remove duplicates
#     unique_playlists = []
#     seen_ids = set()
#     for playlist in all_playlists:
#         if playlist['playlist_id'] not in seen_ids:
#             unique_playlists.append(playlist)
#             seen_ids.add(playlist['playlist_id'])
    
#     return unique_playlists[:max_results]

# def search_playlists_web_fallback(emotion, max_results=8):
#     """Fallback method for playlist search"""
#     playlist_terms = EMOTION_TO_PLAYLIST_TERMS.get(emotion, EMOTION_TO_PLAYLIST_TERMS['Neutral'])
#     playlists = []
    
#     for i, search_term in enumerate(playlist_terms[:max_results]):
#         encoded_term = urllib.parse.quote_plus(search_term)
#         search_url = f"https://www.youtube.com/results?search_query={encoded_term}&sp=EgIQAw%253D%253D"  # Filter for playlists
        
#         playlist_info = {
#             'title': search_term.title(),
#             'channel': 'Various Artists',
#             'playlist_id': f'search_{i}',
#             'url': search_url,
#             'thumbnail': '',
#             'published': 'Various'
#         }
#         playlists.append(playlist_info)
    
#     return playlists

# def parse_duration(duration_str):
#     """Parse YouTube duration format (PT4M13S) to seconds"""
#     import re
#     pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
#     match = re.match(pattern, duration_str)
    
#     if not match:
#         return 0
    
#     hours = int(match.group(1) or 0)
#     minutes = int(match.group(2) or 0)
#     seconds = int(match.group(3) or 0)
    
#     return hours * 3600 + minutes * 60 + seconds

# def format_duration(seconds):
#     """Format seconds to MM:SS"""
#     minutes = seconds // 60
#     seconds = seconds % 60
#     return f"{minutes}:{seconds:02d}"

# def display_artists_menu(artists, emotion):
#     """Display artists menu for selection"""
#     print(f"\n[INFO] Popular {emotion} Artists:")
#     print("=" * 50)
    
#     for i, artist in enumerate(artists, 1):
#         print(f"{i}. {artist}")
    
#     print(f"{len(artists) + 1}. Search for a custom artist")
#     print("0. Go back to main menu")
    
#     return artists

# def display_artist_songs(songs, artist_name):
#     """Display songs by selected artist"""
#     print(f"\n[INFO] Songs by {artist_name} (3-5 minute range):")
#     print("=" * 60)
    
#     if not songs:
#         print("No songs found or using web search mode.")
#         return
    
#     for i, song in enumerate(songs, 1):
#         print(f"{i}. {song['title']}")
#         print(f"   Channel: {song['channel']}")
#         print(f"   Duration: {song['duration']}")
#         print(f"   URL: {song['url']}")
#         print()

# def display_playlists_menu(playlists, emotion):
#     """Display playlists menu for selection"""
#     print(f"\n[INFO] {emotion} Music Playlists:")
#     print("=" * 50)
    
#     for i, playlist in enumerate(playlists, 1):
#         print(f"{i}. {playlist['title']}")
#         print(f"   Channel: {playlist['channel']}")
#         print(f"   URL: {playlist['url']}")
#         print()
    
#     print("0. Go back to main menu")

# def get_user_choice(max_choice, prompt="Enter your choice: "):
#     """Get user choice with validation"""
#     while True:
#         try:
#             choice = int(input(prompt))
#             if 0 <= choice <= max_choice:
#                 return choice
#             else:
#                 print(f"Please enter a number between 0 and {max_choice}")
#         except ValueError:
#             print("Please enter a valid number")

# # Load the trained model and face detector
# model = load_model('model_file.h5')
# faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# # Label dictionary
# labels_dict = {
#     0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
#     4: 'Neutral', 5: 'Sad', 6: 'Surprise'
# }

# print("[INFO] Emotion Detection with Enhanced YouTube Music Integration")
# print("=" * 60)

# # Optional: Set up YouTube API key
# api_setup = input("Do you have a YouTube API key for better results? (y/n): ").lower().strip()
# if api_setup == 'y':
#     api_key = input("Enter your YouTube API key: ").strip()
#     if api_key:
#         YOUTUBE_API_KEY = api_key
#         print("[INFO] YouTube API configured for enhanced features!")
#     else:
#         print("[INFO] Using web search fallback mode.")
# else:
#     print("[INFO] Using web search fallback mode (basic functionality).")

# # Ask user for image path
# image_path = input("\nEnter image path (or press Enter to use random test image): ").strip()

# if not image_path:
#     test_dir = r"C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test"
#     if os.path.exists(test_dir):
#         folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
#         if folders:
#             random_folder = random.choice(folders)
#             images = [f for f in os.listdir(random_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
#             if images:
#                 image_path = os.path.join(random_folder, random.choice(images))
#                 print(f"[INFO] Using random test image: {image_path}")
#             else:
#                 print("[ERROR] No images found in test directory.")
#                 exit()
#         else:
#             print("[ERROR] No emotion folders found in test directory.")
#             exit()
#     else:
#         image_path = input("Test directory not found. Please enter full image path: ").strip()

# # Read image
# frame = cv2.imread(image_path)

# if frame is None:
#     print("[ERROR] Could not load image. Check path.")
#     exit()

# # Resize for better viewing without distortion
# scale_percent = 150  # 150% of original size
# width = int(frame.shape[1] * scale_percent / 100)
# height = int(frame.shape[0] * scale_percent / 100)
# frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

# print(f"\n[INFO] Faces detected: {len(faces)}")

# detected_emotions = []

# if len(faces) == 0:
#     print("[WARNING] No faces found!")
# else:
#     for x, y, w, h in faces:
#         # Predict the emotion
#         face_img = gray[y:y+h, x:x+w]
#         resized = cv2.resize(face_img, (48, 48))
#         normalized = resized / 255.0
#         reshaped = np.reshape(normalized, (1, 48, 48, 1))
#         result = model.predict(reshaped)
#         label = labels_dict[np.argmax(result)]
        
#         detected_emotions.append(label)
        
#         # Draw rectangle around the face
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
#         # Draw label *above* the face rectangle
#         label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#         label_y = max(y - 10, label_size[1] + 10)
#         cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
#         print(f"[INFO] Detected emotion: {label}")

# # Show result
# cv2.imshow("Emotion Detection", frame)

# # Process YouTube integration if emotion detected
# if detected_emotions:
#     # Use the most common emotion if multiple faces
#     primary_emotion = max(set(detected_emotions), key=detected_emotions.count)
    
#     print(f"\n[INFO] Primary emotion detected: {primary_emotion}")
    
#     while True:
#         print(f"\n[MENU] What would you like to listen to for your {primary_emotion} mood?")
#         print("=" * 60)
#         print("1. Single Artist Songs (3-5 minute songs)")
#         print("2. Music Playlists")
#         print("3. Exit")
        
#         main_choice = get_user_choice(3, "Choose option (1-3): ")
        
#         if main_choice == 1:
#             # Single Artist Mode
#             artists = EMOTION_TO_ARTISTS.get(primary_emotion, EMOTION_TO_ARTISTS['Neutral'])
#             display_artists_menu(artists, primary_emotion)
            
#             artist_choice = get_user_choice(len(artists) + 1, f"Choose artist (0-{len(artists) + 1}): ")
            
#             if artist_choice == 0:
#                 continue  # Go back to main menu
#             elif artist_choice == len(artists) + 1:
#                 # Custom artist search
#                 custom_artist = input("Enter artist name: ").strip()
#                 if custom_artist:
#                     selected_artist = custom_artist
#                 else:
#                     continue
#             else:
#                 selected_artist = artists[artist_choice - 1]
            
#             print(f"\n[INFO] Searching for songs by {selected_artist}...")
#             artist_songs = search_artist_songs(selected_artist, primary_emotion)
            
#             display_artist_songs(artist_songs, selected_artist)
            
#             if artist_songs:
#                 print(f"\nOptions for {selected_artist}:")
#                 print("1. Open artist's main YouTube page")
#                 print("2. Open specific search results")
#                 print("3. Go back to menu")
                
#                 open_choice = get_user_choice(3, "Choose option (1-3): ")
                
#                 if open_choice == 1:
#                     # Search for artist's official channel
#                     artist_channel_search = urllib.parse.quote_plus(f"{selected_artist} official")
#                     channel_url = f"https://www.youtube.com/results?search_query={artist_channel_search}&sp=EgIQAg%253D%253D"
#                     webbrowser.open(channel_url)
#                     print(f"[INFO] Opened {selected_artist}'s channel search")
#                 elif open_choice == 2:
#                     if len(artist_songs) == 1 and 'search_' in artist_songs[0]['video_id']:
#                         webbrowser.open(artist_songs[0]['url'])
#                         print(f"[INFO] Opened YouTube search for {selected_artist}")
#                     else:
#                         # Open first song or let user choose
#                         song_choice = get_user_choice(len(artist_songs), f"Choose song to open (1-{len(artist_songs)}): ")
#                         if song_choice > 0:
#                             selected_song = artist_songs[song_choice - 1]
#                             webbrowser.open(selected_song['url'])
#                             print(f"[INFO] Opened: {selected_song['title']}")
#                 elif open_choice == 3:
#                     continue
#             else:
#                 print(f"[WARNING] No songs found for {selected_artist}. Opening general search...")
#                 general_search = urllib.parse.quote_plus(f"{selected_artist} songs")
#                 general_url = f"https://www.youtube.com/results?search_query={general_search}"
#                 webbrowser.open(general_url)
#                 print(f"[INFO] Opened general search for {selected_artist}")
        
#         elif main_choice == 2:
#             # Playlist Mode
#             print(f"\n[INFO] Searching for {primary_emotion} playlists...")
#             playlists = search_playlists(primary_emotion)
            
#             display_playlists_menu(playlists, primary_emotion)
            
#             playlist_choice = get_user_choice(len(playlists), f"Choose playlist (0-{len(playlists)}): ")
            
#             if playlist_choice == 0:
#                 continue  # Go back to main menu
            
#             selected_playlist = playlists[playlist_choice - 1]
            
#             open_choice = input(f"\nOpen '{selected_playlist['title']}'? (y/n): ").lower().strip()
#             if open_choice == 'y':
#                 webbrowser.open(selected_playlist['url'])
#                 print(f"[INFO] Opened playlist: {selected_playlist['title']}")
        
#         elif main_choice == 3:
#             print("[INFO] Exiting music selection...")
#             break

# else:
#     print("\n[INFO] No emotions detected. Cannot recommend music.")

# print("\nPress any key to close the image window...")
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# print("\n[INFO] Program completed successfully!")

#####code with surprise me option######


import cv2
import numpy as np
from keras.models import load_model
import os
import random
import webbrowser
import requests
from datetime import datetime, timedelta
import urllib.parse
import json

# YouTube API Configuration
YOUTUBE_API_KEY = "your_youtube_api_key_here"
YOUTUBE_API_BASE_URL = "https://www.googleapis.com/youtube/v3"

# Emotion to search terms mapping for artists and playlists
EMOTION_TO_ARTISTS = {
    'Happy': ['Ed Sheeran', 'Bruno Mars', 'Dua Lipa', 'The Weeknd', 'Taylor Swift', 'Post Malone'],
    'Sad': ['Billie Eilish', 'Lana Del Rey', 'Adele', 'Sam Smith', 'Lewis Capaldi', 'Olivia Rodrigo'],
    'Angry': ['Eminem', 'Linkin Park', 'Imagine Dragons', 'Twenty One Pilots', 'Metallica', 'Green Day'],
    'Fear': ['Hans Zimmer', 'Trent Reznor', 'John Carpenter', 'Akira Yamaoka', 'Silent Hill OST', 'Horror Music'],
    'Surprise': ['Doja Cat', 'Lil Nas X', 'Bad Bunny', 'BTS', 'Blackpink', 'Trending Artists'],
    'Disgust': ['Arctic Monkeys', 'Radiohead', 'Nirvana', 'Pearl Jam', 'Foo Fighters', 'Red Hot Chili Peppers'],
    'Neutral': ['Lofi Girl', 'Bon Iver', 'Phoebe Bridgers', 'Mac DeMarco', 'Tame Impala', 'Indie Folk']
}

EMOTION_TO_PLAYLIST_TERMS = {
    'Happy': ['happy playlist 2024', 'feel good music playlist', 'upbeat songs playlist', 'pop hits 2024'],
    'Sad': ['sad songs playlist 2024', 'heartbreak playlist', 'emotional music playlist', 'crying songs'],
    'Angry': ['rock playlist 2024', 'metal music playlist', 'aggressive songs playlist', 'workout music'],
    'Fear': ['dark music playlist', 'horror soundtrack playlist', 'scary music playlist', 'thriller music'],
    'Surprise': ['trending playlist 2024', 'viral songs playlist', 'top hits 2024', 'popular music playlist'],
    'Disgust': ['alternative rock playlist', 'indie playlist 2024', 'grunge music playlist', 'underground music'],
    'Neutral': ['chill playlist 2024', 'relaxing music playlist', 'study music playlist', 'ambient playlist']
}
def search_artist_web_fallback(artist_name, emotion, max_results=10):
    """Fallback method for artist search - creates multiple search options"""
    print(f"[INFO] Using web search for {artist_name}")
    
    search_options = [
        f"{artist_name} songs",
        f"{artist_name} hits",
        f"{artist_name} best songs",
        f"{artist_name} official music videos",
        f"{artist_name} top tracks"
    ]
    
    fallback_results = []
    
    for i, search_term in enumerate(search_options[:max_results]):
        encoded_term = urllib.parse.quote_plus(search_term)
        search_url = f"https://www.youtube.com/results?search_query={encoded_term}&sp=EgIQAQ%253D%253D"
        
        song_info = {
            'title': f"{search_term.title()}",
            'channel': 'YouTube Search',
            'video_id': f'search_{i}',
            'url': search_url,
            'duration': 'Various',
            'thumbnail': '',
            'published': 'Recent'
        }
        fallback_results.append(song_info)
    
    return fallback_results
def search_artist_songs(artist_name, emotion, max_results=10):
    """Search for songs by a specific artist with 3-5 minute duration"""
    if YOUTUBE_API_KEY == "your_youtube_api_key_here":
        return search_artist_web_fallback(artist_name, emotion, max_results)
    
    try:
        # Try multiple search strategies
        search_queries = [
            f"{artist_name} official songs",
            f"{artist_name} music videos",
            f"{artist_name} hits",
            f"{artist_name} best songs"
        ]
        
        all_songs = []
        
        for search_query in search_queries[:2]:  # Try first 2 queries
            print(f"[DEBUG] Searching with query: {search_query}")
            
            params = {
                'part': 'snippet',
                'q': search_query,
                'type': 'video',
                'maxResults': 15,  # Get more results
                'order': 'relevance',
                'videoCategoryId': '10',  # Music category
                'key': YOUTUBE_API_KEY
            }
            
            response = requests.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
            print(f"[DEBUG] Search response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                print(f"[DEBUG] Found {len(items)} videos")
                
                if items:
                    video_ids = [item['id']['videoId'] for item in items]
                    
                    # Get video details including duration
                    details_params = {
                        'part': 'snippet,contentDetails',
                        'id': ','.join(video_ids),
                        'key': YOUTUBE_API_KEY
                    }
                    
                    details_response = requests.get(f"{YOUTUBE_API_BASE_URL}/videos", params=details_params)
                    print(f"[DEBUG] Details response status: {details_response.status_code}")
                    
                    if details_response.status_code == 200:
                        details_data = details_response.json()
                        detail_items = details_data.get('items', [])
                        print(f"[DEBUG] Got details for {len(detail_items)} videos")
                        
                        for item in detail_items:
                            try:
                                duration = item['contentDetails']['duration']
                                duration_seconds = parse_duration(duration)
                                
                                # More lenient duration filter: 1-8 minutes (60-480 seconds)
                                if 60 <= duration_seconds <= 480:
                                    title = item['snippet']['title']
                                    channel = item['snippet']['channelTitle']
                                    
                                    # More flexible artist matching
                                    artist_in_title = any(word in title.lower() for word in artist_name.lower().split())
                                    artist_in_channel = any(word in channel.lower() for word in artist_name.lower().split())
                                    
                                    # Include if artist name appears in title or channel
                                    if artist_in_title or artist_in_channel:
                                        song_info = {
                                            'title': title,
                                            'channel': channel,
                                            'video_id': item['id'],
                                            'url': f"https://www.youtube.com/watch?v={item['id']}",
                                            'duration': format_duration(duration_seconds),
                                            'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                                            'published': item['snippet']['publishedAt'][:10]  # Just date
                                        }
                                        all_songs.append(song_info)
                                        print(f"[DEBUG] Added song: {title[:50]}...")
                            except Exception as e:
                                print(f"[DEBUG] Error processing video: {e}")
                                continue
            else:
                print(f"[ERROR] Search API error: {response.status_code}")
                if response.status_code == 403:
                    print("[ERROR] API quota exceeded or invalid key")
                    return search_artist_web_fallback(artist_name, emotion, max_results)
        
        # Remove duplicates based on video_id
        unique_songs = []
        seen_ids = set()
        for song in all_songs:
            if song['video_id'] not in seen_ids:
                unique_songs.append(song)
                seen_ids.add(song['video_id'])
        
        print(f"[DEBUG] Total unique songs found: {len(unique_songs)}")
        
        if unique_songs:
            return unique_songs[:max_results]
        else:
            print("[DEBUG] No songs found with API, using web fallback")
            return search_artist_web_fallback(artist_name, emotion, max_results)
    
    except Exception as e:
        print(f"[ERROR] Artist search failed: {e}")
        return search_artist_web_fallback(artist_name, emotion, max_results)

def search_surprise_songs(emotion, max_results=10):
    """Search for surprise songs based on emotion - mix of popular and recent"""
    if YOUTUBE_API_KEY == "your_youtube_api_key_here":
        return search_surprise_web_fallback(emotion, max_results)
    
    # Create diverse search terms for surprise songs
    surprise_search_terms = {
        'Happy': [
            'feel good songs ', 'upbeat hits 2024', 'happy pop songs', 
            'dance music ', 'positive vibes songs', 'summer hits 2024'
        ],
        'Sad': [
            'sad songs 2024', 'emotional ballads', 'heartbreak songs', 
            'melancholy music', 'acoustic sad songs', 'crying songs 2024'
        ],
        'Angry': [
            'rock songs 2024', 'aggressive music', 'metal hits', 
            'angry rap songs', 'hardcore music', 'intense songs 2024'
        ],
        'Fear': [
            'dark music', 'scary songs', 'horror soundtrack', 
            'intense music', 'thriller music', 'eerie songs'
        ],
        'Surprise': [
            'trending songs 2024', 'viral hits', 'unexpected songs', 
            'surprise hits', 'random popular songs', 'discovery music'
        ],
        'Disgust': [
            'alternative rock 2024', 'indie songs', 'underground music', 
            'grunge songs', 'experimental music', 'unique songs'
        ],
        'Neutral': [
            'chill songs 2024', 'relaxing music', 'background music', 
            'ambient songs', 'study music', 'calm songs'
        ]
    }
    
    search_terms = surprise_search_terms.get(emotion, surprise_search_terms['Neutral'])
    all_songs = []
    
    try:
        # Use multiple search terms to get variety
        for search_term in search_terms[:3]:  # Use first 3 terms
            print(f"[DEBUG] Surprise search with: {search_term}")
            
            params = {
                'part': 'snippet',
                'q': search_term,
                'type': 'video',
                'maxResults': 5,
                'order': 'relevance',
                'videoCategoryId': '10',  # Music category
                'key': YOUTUBE_API_KEY
            }
            
            response = requests.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get('items', [])
                
                if items:
                    video_ids = [item['id']['videoId'] for item in items]
                    
                    # Get video details
                    details_params = {
                        'part': 'snippet,contentDetails',
                        'id': ','.join(video_ids),
                        'key': YOUTUBE_API_KEY
                    }
                    
                    details_response = requests.get(f"{YOUTUBE_API_BASE_URL}/videos", params=details_params)
                    
                    if details_response.status_code == 200:
                        details_data = details_response.json()
                        
                        for item in details_data.get('items', []):
                            try:
                                duration = item['contentDetails']['duration']
                                duration_seconds = parse_duration(duration)
                                
                                # Filter for songs between 1-8 minutes
                                if 60 <= duration_seconds <= 480:
                                    song_info = {
                                        'title': item['snippet']['title'],
                                        'channel': item['snippet']['channelTitle'],
                                        'video_id': item['id'],
                                        'url': f"https://www.youtube.com/watch?v={item['id']}",
                                        'duration': format_duration(duration_seconds),
                                        'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                                        'published': item['snippet']['publishedAt'][:10]
                                    }
                                    all_songs.append(song_info)
                            except Exception as e:
                                continue
        
        # Remove duplicates and shuffle for surprise effect
        unique_songs = []
        seen_ids = set()
        for song in all_songs:
            if song['video_id'] not in seen_ids:
                unique_songs.append(song)
                seen_ids.add(song['video_id'])
        
        # Shuffle the songs for surprise
        import random as rand
        rand.shuffle(unique_songs)
        
        print(f"[DEBUG] Found {len(unique_songs)} surprise songs")
        return unique_songs[:max_results]
    
    except Exception as e:
        print(f"[ERROR] Surprise search failed: {e}")
        return search_surprise_web_fallback(emotion, max_results)

def search_surprise_web_fallback(emotion, max_results=10):
    """Fallback surprise search using web URLs"""
    surprise_terms = [
        f"{emotion} songs 2024",
        f"best {emotion} music",
        f"{emotion} hits playlist",
        f"popular {emotion} songs",
        f"{emotion} music mix",
        f"trending {emotion} music",
        f"{emotion} songs you need to hear",
        f"amazing {emotion} songs",
        f"top {emotion} tracks",
        f"{emotion} music discovery"
    ]
    
    surprise_songs = []
    
    for i, search_term in enumerate(surprise_terms[:max_results]):
        encoded_term = urllib.parse.quote_plus(search_term)
        search_url = f"https://www.youtube.com/results?search_query={encoded_term}&sp=EgIQAQ%253D%253D"
        
        song_info = {
            'title': f"Surprise {emotion} Song #{i+1}",
            'channel': search_term.title(),
            'video_id': f'surprise_{i}',
            'url': search_url,
            'duration': 'Various',
            'thumbnail': '',
            'published': 'Recent'
        }
        surprise_songs.append(song_info)
    
    return surprise_songs

def display_surprise_songs(songs, emotion):
    """Display surprise songs with numbers"""
    print(f"\n[INFO] 🎵 Surprise {emotion} Songs Just For You! 🎵")
    print("=" * 60)
    
    if not songs:
        print("Using web search mode - each option will take you to YouTube search.")
        print()
    
    for i, song in enumerate(songs, 1):
        print(f"{i:2d}. {song['title']}")
        print(f"     Channel: {song['channel']}")
        if song['duration'] != 'Various':
            print(f"     Duration: {song['duration']}")
        print(f"     URL: {song['url']}")
        print()
    
    print(f"[INFO] Found {len(songs)} surprise songs for your {emotion} mood!")

def search_playlists(emotion, max_results=8):
    """Search for playlists based on emotion"""
    if YOUTUBE_API_KEY == "your_youtube_api_key_here":
        return search_playlists_web_fallback(emotion, max_results)
    
    playlist_terms = EMOTION_TO_PLAYLIST_TERMS.get(emotion, EMOTION_TO_PLAYLIST_TERMS['Neutral'])
    all_playlists = []
    
    try:
        for search_term in playlist_terms[:3]:  # Use first 3 terms
            params = {
                'part': 'snippet',
                'q': search_term,
                'type': 'playlist',
                'maxResults': 3,
                'order': 'relevance',
                'key': YOUTUBE_API_KEY
            }
            
            response = requests.get(f"{YOUTUBE_API_BASE_URL}/search", params=params)
            
            if response.status_code == 200:
                data = response.json()
                for item in data.get('items', []):
                    playlist_info = {
                        'title': item['snippet']['title'],
                        'channel': item['snippet']['channelTitle'],
                        'playlist_id': item['id']['playlistId'],
                        'url': f"https://www.youtube.com/playlist?list={item['id']['playlistId']}",
                        'thumbnail': item['snippet']['thumbnails']['medium']['url'],
                        'published': item['snippet']['publishedAt']
                    }
                    all_playlists.append(playlist_info)
    
    except Exception as e:
        print(f"[ERROR] Playlist search failed: {e}")
        return search_playlists_web_fallback(emotion, max_results)
    
    # Remove duplicates
    unique_playlists = []
    seen_ids = set()
    for playlist in all_playlists:
        if playlist['playlist_id'] not in seen_ids:
            unique_playlists.append(playlist)
            seen_ids.add(playlist['playlist_id'])
    
    return unique_playlists[:max_results]

def search_playlists_web_fallback(emotion, max_results=8):
    """Fallback method for playlist search"""
    playlist_terms = EMOTION_TO_PLAYLIST_TERMS.get(emotion, EMOTION_TO_PLAYLIST_TERMS['Neutral'])
    playlists = []
    
    for i, search_term in enumerate(playlist_terms[:max_results]):
        encoded_term = urllib.parse.quote_plus(search_term)
        search_url = f"https://www.youtube.com/results?search_query={encoded_term}&sp=EgIQAw%253D%253D"  # Filter for playlists
        
        playlist_info = {
            'title': search_term.title(),
            'channel': 'Various Artists',
            'playlist_id': f'search_{i}',
            'url': search_url,
            'thumbnail': '',
            'published': 'Various'
        }
        playlists.append(playlist_info)
    
    return playlists

def parse_duration(duration_str):
    """Parse YouTube duration format (PT4M13S) to seconds"""
    import re
    pattern = r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?'
    match = re.match(pattern, duration_str)
    
    if not match:
        return 0
    
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = int(match.group(3) or 0)
    
    return hours * 3600 + minutes * 60 + seconds

def format_duration(seconds):
    """Format seconds to MM:SS"""
    minutes = seconds // 60
    seconds = seconds % 60
    return f"{minutes}:{seconds:02d}"

def display_artists_menu(artists, emotion):
    """Display artists menu for selection"""
    print(f"\n[INFO] Popular {emotion} Artists:")
    print("=" * 50)
    
    for i, artist in enumerate(artists, 1):
        print(f"{i}. {artist}")
    
    print(f"{len(artists) + 1}. Search for a custom artist")
    print("0. Go back to main menu")
    
    return artists

def display_artist_songs(songs, artist_name):
    """Display songs by selected artist"""
    print(f"\n[INFO] Songs by {artist_name} (3-5 minute range):")
    print("=" * 60)
    
    if not songs:
        print("No songs found or using web search mode.")
        return
    
    for i, song in enumerate(songs, 1):
        print(f"{i}. {song['title']}")
        print(f"   Channel: {song['channel']}")
        print(f"   Duration: {song['duration']}")
        print(f"   URL: {song['url']}")
        print()

def display_playlists_menu(playlists, emotion):
    """Display playlists menu for selection"""
    print(f"\n[INFO] {emotion} Music Playlists:")
    print("=" * 50)
    
    for i, playlist in enumerate(playlists, 1):
        print(f"{i}. {playlist['title']}")
        print(f"   Channel: {playlist['channel']}")
        print(f"   URL: {playlist['url']}")
        print()
    
    print("0. Go back to main menu")

def get_user_choice(max_choice, prompt="Enter your choice: "):
    """Get user choice with validation"""
    while True:
        try:
            choice = int(input(prompt))
            if 0 <= choice <= max_choice:
                return choice
            else:
                print(f"Please enter a number between 0 and {max_choice}")
        except ValueError:
            print("Please enter a valid number")

# Load the trained model and face detector
model = load_model('model_file.h5')
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Label dictionary
labels_dict = {
    0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy',
    4: 'Neutral', 5: 'Sad', 6: 'Surprise'
}

print("[INFO] Emotion Detection with Enhanced YouTube Music Integration")
print("=" * 60)

# Optional: Set up YouTube API key
api_setup = input("Do you have a YouTube API key for better results? (y/n): ").lower().strip()
if api_setup == 'y':
    api_key = input("Enter your YouTube API key: ").strip()
    if api_key:
        YOUTUBE_API_KEY = api_key
        print("[INFO] YouTube API configured for enhanced features!")
    else:
        print("[INFO] Using web search fallback mode.")
else:
    print("[INFO] Using web search fallback mode (basic functionality).")

# Ask user for image path
image_path = input("\nEnter image path (or press Enter to use random test image): ").strip()

if not image_path:
    test_dir = r"C:\Users\MANASVI\Documents\GitHub\Emotion_detection\test"
    if os.path.exists(test_dir):
        folders = [os.path.join(test_dir, f) for f in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, f))]
        if folders:
            random_folder = random.choice(folders)
            images = [f for f in os.listdir(random_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
            if images:
                image_path = os.path.join(random_folder, random.choice(images))
                print(f"[INFO] Using random test image: {image_path}")
            else:
                print("[ERROR] No images found in test directory.")
                exit()
        else:
            print("[ERROR] No emotion folders found in test directory.")
            exit()
    else:
        image_path = input("Test directory not found. Please enter full image path: ").strip()

# Read image
frame = cv2.imread(image_path)

if frame is None:
    print("[ERROR] Could not load image. Check path.")
    exit()

# Resize for better viewing without distortion
scale_percent = 150  # 150% of original size
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

print(f"\n[INFO] Faces detected: {len(faces)}")

detected_emotions = []

if len(faces) == 0:
    print("[WARNING] No faces found!")
else:
    for x, y, w, h in faces:
        # Predict the emotion
        face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(face_img, (48, 48))
        normalized = resized / 255.0
        reshaped = np.reshape(normalized, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = labels_dict[np.argmax(result)]
        
        detected_emotions.append(label)
        
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        
        # Draw label *above* the face rectangle
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        label_y = max(y - 10, label_size[1] + 10)
        cv2.putText(frame, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        print(f"[INFO] Detected emotion: {label}")

# Show result
cv2.imshow("Emotion Detection", frame)

# Process YouTube integration if emotion detected
if detected_emotions:
    # Use the most common emotion if multiple faces
    primary_emotion = max(set(detected_emotions), key=detected_emotions.count)
    
    print(f"\n[INFO] Primary emotion detected: {primary_emotion}")
    
    while True:
        print(f"\n[MENU] What would you like to listen to for your {primary_emotion} mood?")
        print("=" * 60)
        print("1. Single Artist Songs (3-5 minute songs)")
        print("2. Music Playlists")
        print("3. Exit")
        
        main_choice = get_user_choice(3, "Choose option (1-3): ")
        
        if main_choice == 1:
            # Single Artist Mode - Show sub-menu
            print(f"\n[SUBMENU] Single Songs for your {primary_emotion} mood:")
            print("=" * 50)
            print("1. Surprise Me (10 random songs)")
            print("2. Specific Artist")
            print("0. Go back to main menu")
            
            song_choice = get_user_choice(2, "Choose option (0-2): ")
            
            if song_choice == 0:
                continue  # Go back to main menu
            elif song_choice == 1:
                # Surprise Me Mode
                print(f"\n[INFO] Finding surprise {primary_emotion} songs for you...")
                surprise_songs = search_surprise_songs(primary_emotion, 10)
                
                display_surprise_songs(surprise_songs, primary_emotion)
                
                if surprise_songs:
                    print(f"\nWhat would you like to do with these {primary_emotion} songs?")
                    print("1. Play first song")
                    print("2. Choose a specific song to play")
                    print("3. Open all songs in separate tabs")
                    print("4. Go back to menu")
                    
                    surprise_action = get_user_choice(4, "Choose action (1-4): ")
                    
                    if surprise_action == 1:
                        webbrowser.open(surprise_songs[0]['url'])
                        print(f"[INFO] Opened: {surprise_songs[0]['title']}")
                    elif surprise_action == 2:
                        song_select = get_user_choice(len(surprise_songs), f"Choose song (1-{len(surprise_songs)}): ")
                        if song_select > 0:
                            selected_song = surprise_songs[song_select - 1]
                            webbrowser.open(selected_song['url'])
                            print(f"[INFO] Opened: {selected_song['title']}")
                    elif surprise_action == 3:
                        for song in surprise_songs[:5]:  # Limit to 5 tabs to avoid overwhelming
                            webbrowser.open(song['url'])
                        print(f"[INFO] Opened first 5 surprise songs in browser tabs")
                    elif surprise_action == 4:
                        continue
                else:
                    # Fallback if no songs found
                    print(f"[INFO] Creating fallback surprise search...")
                    fallback_search = urllib.parse.quote_plus(f"{primary_emotion} songs 2024 hits")
                    fallback_url = f"https://www.youtube.com/results?search_query={fallback_search}"
                    webbrowser.open(fallback_url)
                    print(f"[INFO] Opened surprise {primary_emotion} songs search")
            
            elif song_choice == 2:
                # Specific Artist Mode (existing functionality)
                artists = EMOTION_TO_ARTISTS.get(primary_emotion, EMOTION_TO_ARTISTS['Neutral'])
                display_artists_menu(artists, primary_emotion)
                
                artist_choice = get_user_choice(len(artists) + 1, f"Choose artist (0-{len(artists) + 1}): ")
                
                if artist_choice == 0:
                    continue  # Go back to main menu
                elif artist_choice == len(artists) + 1:
                    # Custom artist search
                    custom_artist = input("Enter artist name: ").strip()
                    if custom_artist:
                        selected_artist = custom_artist
                    else:
                        continue
                else:
                    selected_artist = artists[artist_choice - 1]
                
                print(f"\n[INFO] Searching for songs by {selected_artist}...")
                artist_songs = search_artist_songs(selected_artist, primary_emotion)
                
                display_artist_songs(artist_songs, selected_artist)
                
                if artist_songs:
                    print(f"\nOptions for {selected_artist}:")
                    print("1. Open artist's main YouTube page")
                    print("2. Open specific search results")
                    print("3. Go back to menu")
                    
                    open_choice = get_user_choice(3, "Choose option (1-3): ")
                    
                    if open_choice == 1:
                        # Search for artist's official channel
                        artist_channel_search = urllib.parse.quote_plus(f"{selected_artist} official")
                        channel_url = f"https://www.youtube.com/results?search_query={artist_channel_search}&sp=EgIQAg%253D%253D"
                        webbrowser.open(channel_url)
                        print(f"[INFO] Opened {selected_artist}'s channel search")
                    elif open_choice == 2:
                        if len(artist_songs) == 1 and 'search_' in artist_songs[0]['video_id']:
                            webbrowser.open(artist_songs[0]['url'])
                            print(f"[INFO] Opened YouTube search for {selected_artist}")
                        else:
                            # Open first song or let user choose
                            song_choice = get_user_choice(len(artist_songs), f"Choose song to open (1-{len(artist_songs)}): ")
                            if song_choice > 0:
                                selected_song = artist_songs[song_choice - 1]
                                webbrowser.open(selected_song['url'])
                                print(f"[INFO] Opened: {selected_song['title']}")
                    elif open_choice == 3:
                        continue
                else:
                    print(f"[WARNING] No songs found for {selected_artist}. Opening general search...")
                    general_search = urllib.parse.quote_plus(f"{selected_artist} songs")
                    general_url = f"https://www.youtube.com/results?search_query={general_search}"
                    webbrowser.open(general_url)
                    print(f"[INFO] Opened general search for {selected_artist}")
        
        elif main_choice == 2:
            # Playlist Mode
            print(f"\n[INFO] Searching for {primary_emotion} playlists...")
            playlists = search_playlists(primary_emotion)
            
            display_playlists_menu(playlists, primary_emotion)
            
            playlist_choice = get_user_choice(len(playlists), f"Choose playlist (0-{len(playlists)}): ")
            
            if playlist_choice == 0:
                continue  # Go back to main menu
            
            selected_playlist = playlists[playlist_choice - 1]
            
            open_choice = input(f"\nOpen '{selected_playlist['title']}'? (y/n): ").lower().strip()
            if open_choice == 'y':
                webbrowser.open(selected_playlist['url'])
                print(f"[INFO] Opened playlist: {selected_playlist['title']}")
        
        elif main_choice == 3:
            print("[INFO] Exiting music selection...")
            break

else:
    print("\n[INFO] No emotions detected. Cannot recommend music.")

print("\nPress any key to close the image window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
print("\n[INFO] Program completed successfully!")