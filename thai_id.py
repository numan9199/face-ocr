from ThaiPersonalCardExtract import PersonalCard
reader = PersonalCard(lang="mix", tesseract_cmd="C:\Program Files\Tesseract-OCR") # for windows need to pass tesseract_cmd parameter to setup your tesseract command path.
result = reader.extract_front_info('thai id.jpeg')
print(result)