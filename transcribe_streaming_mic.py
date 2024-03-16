# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Google Cloud Speech API sample application using the streaming API.

NOTE: This module requires the additional dependency pyaudio. To install
using pip:

    pip install pyaudio

Example usage:
    python transcribe_streaming_mic.py
"""

# [START speech_transcribe_streaming_mic]

import queue
import re
import sys
import json
import cv2
from google.cloud import speech

import pyaudio
import text2video

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms


class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int = RATE, chunk: int = CHUNK) -> None:
        """The audio -- and generator -- is guaranteed to be on the main thread."""
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self: object) -> object:
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        """Closes the stream, regardless of whether the connection was lost or not."""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
            in_data: The audio data as a bytes object
            frame_count: The number of frames captured
            time_info: The time information
            status_flags: The status flags

        Returns:
            The audio data as a bytes object
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Generates audio chunks from the stream of audio data in chunks.

        Args:
            self: The MicrophoneStream object

        Returns:
            A generator that outputs audio chunks.
        """
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
    
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

# def listen_print_loop(responses):
#     for response in responses:
#         # Check if we have interim results
#         if not response.results:
#             continue

#         result = response.results[0]
#         if not result.alternatives:
#             continue

#         # Display the transcription of the interim result
#         print(result.alternatives[0].transcript)
display_width = 720
display_height = 1280
def add_videos (video_labels):
    videos = [(word,rf"C:\Users\Owner\Documents\FMI\Hackaton\videos\{video}.mp4") for (word,video) in video_labels]
    for i in videos:
        print ("path: ",i)
        (text, path) = i
        video = cv2.VideoCapture(path)
        #video = cv2.VideoCapture(rf"C:\Users\Owner\anaconda3\envs\art10\Lib\site-packages\sign_language_translator\assets\{i}.mp4")
        while True:
            ret, frame = video.read()
            if ret:
                frame = cv2.resize(frame, (display_width, display_height))
                cv2.imshow('Video', frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        video.release()

def listen_print_loop(responses: object, words) -> str:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.

    Args:
        responses: List of server responses

    Returns:
        The transcribed text.
    """
    num_chars_printed = 0
    num_chars = 0
    last_sec = 0
    
    for response in responses:
        if not response.results:
            continue

        # The results list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's is_final, it
        # moves on to considering the next utterance.
        result = response.results[0]
      
        
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            #sys.stdout.write(transcript + overwrite_chars + "\n")
            #sys.stdout.flush()
            #print ("transcript : ", transcript[num_chars_printed:] + overwrite_chars + "\n")

            num_chars_printed = len(transcript)
            
            # if there are 2 seconds form the last print
            if (result.result_end_time.seconds - last_sec) > 1.5:
                
                new_sentence = transcript[num_chars:] + overwrite_chars
                
                video_labels = text2video.get_video_from_text(new_sentence, words)
                
                add_videos (video_labels)
                last_sec = result.result_end_time.seconds
                num_chars = num_chars_printed

        else:
            #print(transcript[num_chars] + overwrite_chars + "\n")
            print('final')
            pass

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0

    
    return transcript


def main() -> None:
    """Transcribe speech from audio file."""
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "bg-BG"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
        
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
            )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)
        file_path = rf"C:\Users\Owner\anaconda3\envs\art10\Lib\site-packages\sign_language_translator\assets\pk-dictionary-mapping.json"
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        words = data[0]['mapping']

        k = 0.5
        display_width = round (720 * k)
        display_height = round (1280 * k)
        flag = True


        # Now, put the transcription responses to use.
       
        listen_print_loop(responses, words)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
# [END speech_transcribe_streaming_mic]