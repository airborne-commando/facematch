import base64
import hashlib
import requests
from io import BytesIO
from PIL import Image
import face_recognition
import numpy as np
import sys

# Will require GPUs

def get_image_bytes(source):
    """Accepts data URI or URL and returns image bytes."""
    if source.startswith('data:'):
        b64_data = source.split(',', 1)[1]
        return base64.b64decode(b64_data)
    else:
        response = requests.get(source)
        response.raise_for_status()
        return response.content

def compute_face_encoding(image_bytes):
    """Extract face encoding from image bytes."""
    try:
        image = Image.open(BytesIO(image_bytes))
        rgb_image = np.array(image.convert('RGB'))
        
        face_encodings = face_recognition.face_encodings(rgb_image)
        if len(face_encodings) == 0:
            return None
        return face_encodings[0]  # Return first face found
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def face_similarity(source1, source2, threshold=0.6):
    """
    Compare face similarity between two image sources.
    Returns similarity score (0-1, higher = more similar) and match boolean.
    """
    bytes1 = get_image_bytes(source1)
    bytes2 = get_image_bytes(source2)
    
    encoding1 = compute_face_encoding(bytes1)
    encoding2 = compute_face_encoding(bytes2)
    
    if encoding1 is None or encoding2 is None:
        return 0.0, False, "No face detected in one or both images"
    
    # Distance metric (lower = more similar)
    distance = face_recognition.face_distance([encoding1], encoding2)[0]
    similarity = 1 - distance  # Convert to similarity score
    
    match = distance < threshold
    return similarity, match, f"Distance: {distance:.3f}"

def compare_multiple_faces(target_source, candidate_sources, threshold=0.6):
    """Compare target face against multiple candidates."""
    target_bytes = get_image_bytes(target_source)
    target_encoding = compute_face_encoding(target_bytes)
    
    if target_encoding is None:
        return None, "No face detected in target image"
    
    results = []
    best_match = None
    best_score = 0
    
    for i, candidate_source in enumerate(candidate_sources):
        candidate_bytes = get_image_bytes(candidate_source)
        candidate_encoding = compute_face_encoding(candidate_bytes)
        
        if candidate_encoding is None:
            results.append((i, 0.0, False, "No face detected"))
            continue
        
        distance = face_recognition.face_distance([target_encoding], candidate_encoding)[0]
        similarity = 1 - distance
        match = distance < threshold
        
        results.append((i, similarity, match, f"Distance: {distance:.3f}"))
        
        if similarity > best_score:
            best_score = similarity
            best_match = i
    
    return best_match, results

# Installation check and instructions
def check_installation():
    """Check if face_recognition is properly installed."""
    try:
        import face_recognition
        print("✓ face_recognition imported successfully")
        return True
    except ImportError as e:
        if "face_recognition_models" in str(e):
            print("❌ Missing face_recognition_models")
            print("💡 Install with: pip install git+https://github.com/ageitgey/face_recognition_models")
        else:
            print("❌ face_recognition not installed")
            print("💡 Install with: pip install face_recognition")
        return False
    except Exception as e:
        print(f"❌ Installation error: {e}")
        return False

if __name__ == "__main__":
    print("🔍 Face Similarity Detection Script")
    print("=" * 50)
    
    # Check installation first
    if not check_installation():
        print("\nPlease fix installation issues before continuing.")
        sys.exit(1)
    
    print("\n📋 Usage Examples:")
    print("1. Compare two images:")
    print("   sim, match, info = face_similarity('url1', 'url2')")
    print("2. Find best match from candidates:")
    print("   best, results = compare_multiple_faces('target.jpg', ['img1.jpg', 'img2.jpg'])")
    
    # Example usage (uncomment and modify paths)
    
    # Test with URLs or data URIs, replace the URLs/URI here.
    target = "https://imageio.forbes.com/specials-images/imageserve/66f5b8cbf6d5e9f3f3703478/1x1-Gabe-Newell-credit-Edge-Magazine-getty-images/0x0.jpg?format=jpg&height=1080&width=1080"
    candidates = [
    "https://static.wikia.nocookie.net/half-life/images/6/62/Gaben.jpg/revision/latest/scale-to-width-down/1200?cb=20200126040848&path-prefix=en",
    "https://sm.ign.com/ign_nordic/news/v/valve-boss/valve-boss-gabe-newell-says-hes-been-retired-in-a-sense-for_n2vz.jpg"
    ]
    
    # Single comparison
    sim, match, info = face_similarity(target, candidates[0])
    print(f"\n✅ Similarity: {sim:.3f}, Match: {match} [{info}]")

    # Multiple comparison
    best_idx, results = compare_multiple_faces(target, candidates)
    if best_idx is None:
        print("\n🏆 No valid candidate (no faces detected).")
    else:
        best_sim = results[best_idx][1]
        best_match_flag = results[best_idx][2]
        print(f"\n🏆 Best match: candidate {best_idx}, similarity={best_sim:.3f}, match={best_match_flag}")
    for idx, sim, match, info in results:
        status = "MATCH" if match else "NO MATCH"
        print(f"Candidate {idx}: match={match} ({status}), similarity={sim:.3f}, {info}")

