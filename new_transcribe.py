import os
import threading
import cv2
import queue
from google.cloud import speech
import pyaudio
import threading
import queue
import math

# Assuming you have set GOOGLE_APPLICATION_CREDENTIALS environment variable
client = speech.SpeechClient()

RATE = 16000
CHUNK = int(RATE / 10)  # Adjust as needed

# Example dictionary mapping words to video file paths
word_to_video = {
    "hello": rf"C:\Users\Owner\Documents\FMI\Hackaton\videos\32581942.mp4",
    "world": rf"C:\Users\Owner\Documents\FMI\Hackaton\videos\32581940.mp4"
    # Add more mappings as needed
}

# Queue for communication between speech recognition and video playback threads
transcript_queue = queue.Queue()

delay = 10
def play_video(video_path):
    """Function to play a video at a faster speed by skipping frames."""
    cap = cv2.VideoCapture(video_path)
    frame_skip = 2  # Define how many frames to skip

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Display every n-th frame (where n is frame_skip + 1)
        if int(cap.get(cv2.CAP_PROP_POS_FRAMES)) % (frame_skip + 1) == 0:
            cv2.imshow("Video", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Minimize waitKey for faster response
                break

    cap.release()

def video_player():
    """Thread function for playing videos based on received transcripts."""
    while True:
        word = transcript_queue.get()  # Blocking get from the queue
        print (word)
        if word in word_to_video:
            play_video(word_to_video["hello"])
        else:
            play_video(word_to_video["world"])
            

def microphone_stream():
    """Generator function to yield audio chunks from the microphone."""
    p = pyaudio.PyAudio()
    stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    try:
        while True:
            yield stream.read(CHUNK)
    finally:
        stream.stop_stream()
        stream.close()
        p.terminate()

def speech_recognizer():
    """Thread function for continuous speech recognition."""
    audio_stream = microphone_stream()
    requests = (speech.StreamingRecognizeRequest(audio_content=chunk) for chunk in audio_stream)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code='bg-BG',
    )
    streaming_config = speech.StreamingRecognitionConfig(config=config, interim_results=True)

    responses = client.streaming_recognize(streaming_config, requests)
    print ("predi def handle_responses")

    # Assuming `responses` is a generator for streaming data
    response_queue = queue.Queue()
    def handle_responses(responses):
        for response in responses:
            # This will block waiting for the next response, but it's in a separate thread
            response_queue.put(response)
    print ("predi thread")
    # Start the response handler in a separate thread
    threading.Thread(target=handle_responses, args=(responses,), daemon=True).start()

    last_processed_pos = 0  # Track the last processed position in the interim transcript
    already_pushed = 0
    rest = []
    while True:
        try:
            # Non-blocking check
            response = response_queue.get(block=False)

            for result in response.results:
                transcript = result.alternatives[0].transcript.lower().strip()

                if not result.is_final and transcript_queue.qsize() < 3:
                    # Process only the new part of the interim transcript since the last update
                    new_part = transcript[last_processed_pos:]
                    new_words = new_part.strip().split()
                    for i in range(3):
                        if already_pushed < len(new_words):
                            transcript_queue.put(new_words[already_pushed])
                            already_pushed += 1
                    rest = []
                    for i in range (already_pushed, len(new_words)):
                        rest.append(new_words[i])
                    #print ("added", rest)
                elif result.is_final:
                    # If the result is final, process the entire transcript
                    # and reset tracking for the next segment of speech
                    #print ("final: " , transcript)
                    final_words = transcript.split()
                    for i in range (already_pushed, len(final_words)):
                        word = final_words[i]
                        transcript_queue.put(word)
                    last_processed_pos = 0  # Reset for the next segment of speech
                    already_pushed = 0
                else:
                    rest = []
                    new_part = transcript[last_processed_pos:]
                    new_words = new_part.strip().split()
                    for i in range (already_pushed, len(new_words)):
                        rest.append(new_words[i])
                    #print ("added 2", rest)
        except queue.Empty:
            # No response available yet
            #print("Response is None")
            #print ("rest: ", rest)
            if transcript_queue.qsize() < 3:
                for i in range (3):
                    if len(rest) > 0:
                        transcript_queue.put(rest[0])
                        rest = rest[1:]
                        already_pushed += 1
                
            

def main():
    # Start the video player thread
    player_thread = threading.Thread(target=video_player, daemon=True)
    player_thread.start()

    # Start the speech recognition in the main thread
    print ("Starting speech recognition...")
    speech_recognizer()

if __name__ == '__main__':
    main()
