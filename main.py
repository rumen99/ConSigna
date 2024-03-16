import sign_language_translator as slt

# print(slt.ModelCodes)
# model = slt.get_model("transformer-text-to-sign")

#model = slt.models.ConcatenativeSynthesis(
#   text_language = "english", # or object of any child of slt.languages.text.text_language.TextLanguage class
#   sign_language = "pakistan-sign-language", # or object of any child of slt.languages.sign.sign_language.SignLanguage class
#   sign_format = "video" # or object of any child of slt.vision.sign.Sign class
#)

embedding_model = slt.models.MediaPipeLandmarksModel()
sign = slt.Video("video.mp4")
embedding = embedding_model.embed(sign.iter_frames())
sign.show()

# sign_language_sentence = model.translate("Hello, how are you?")
# sign_language_sentence.show()


# sign_language_sentence.save("output.mp4")