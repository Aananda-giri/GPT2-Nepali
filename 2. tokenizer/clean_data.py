
# code from ../pre-processing/2. pre-process.ipynb
# Probably should `clean_data(new_crawled_data)` before merging `new_crawled_data.csv` and previous`cleaned_data.csv`

from bs4 import BeautifulSoup
import csv
import json
import re
import string
# import pandas as pd

# from drive.MyDrive.Research.datasets.scrapy_engine.code.load_data import load_data
class CleanData:
    def __init__(self):
        pass


        # return text
    # Example of removing HTML tags
    def clean_html(self, text):
        '''
        # HTML Tag Removal:
        * removes html tags like: <h1>
        * Removes css or js code inside <style> and <script> tags
        '''
        soup = BeautifulSoup(text, "lxml")

        # Remove all <script> and <style> tags
        for script_or_style in soup(['script', 'style']):
            script_or_style.decompose()

        # Get text from the modified HTML
        text = soup.get_text(separator=' ', strip=True)
        # print(text)

        return text

    def convert_to_devanagari_digits(self, input_string):
        # Function to convert 0-9 to ० - ९
        # i.e. Mapping of ASCII digits to Devanagari digits
        devanagari_digits = {
            '0': '०',
            '1': '१',
            '2': '२',
            '3': '३',
            '4': '४',
            '5': '५',
            '6': '६',
            '7': '७',
            '8': '८',
            '9': '९'
        }
        # Convert each digit in the input string
        result = ''.join(devanagari_digits[char] if char in devanagari_digits else char for char in input_string)
        return result

    def remove_non_devanagari_characters(self, text, keep_special_characters=True):
        '''
            # Function to find nepali sequences.
            * keep punctuations if they occur between devanagari characters.
            * Remove punctuation if previous character is not devanagari.
            # Examples
            texts = [
                "उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। \"hi there\". what is your name? उनले दुहेको दूध",
                "\"hi there. \"उनले दुहेको\" दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। hi there. what is your name? उनले दुहेको दूध\"",
                "name? उनले दुहेको दूध\""    #output: (last quatation, name?) should be ignored
                ]

            for text in texts:
                removed = remove_non_devanagari_characters(text)
                print(f'text: {text}, \nclen: {removed}\n\n')


            # output
            text: उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। "hi there". what is your name? उनले दुहेको दूध,
            clen: उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे।             उनले दुहेको दूध


            text: "hi there. "उनले दुहेको" दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। hi there. what is your name? उनले दुहेको दूध",
            clen:    "उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे।             उनले दुहेको दूध"


            text: name? उनले दुहेको दूध",
            clen:  उनले दुहेको दूध"
        '''
        def is_devanagari(char):
            pattern=r'[ऀ-ॿ]'
            return bool(re.match(pattern, char))

        if not keep_special_characters:
            return re.sub(r"[^ऀ-ॿ ]", " ", text)

        sequences = []
        sequence = ''
        punctuation_symbols = string.punctuation    # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        prefix_punctuations = '\"\'(<[{'
        index=0
        while index < len(text):
            char = text[index]
            if is_devanagari(char) or char == ' ':
                # Character is devanagari
                sequence += char
            elif char in punctuation_symbols:
                # Character is punctuation
                if sequence != '':
                    if (len(text) > index+1) and not is_devanagari(text[index+1]):
                        # e.g. गरे। "hi there" : skip quotation before hi
                        pass
                    else:
                        sequence += char    # Sequence is no empty. i.e. previous char/sequence was devanagari otherwise ignore  punctuation
                elif (len(text) > index+1) and is_devanagari(text[index+1]):
                    # preserve prefix punctuations in devanagari. e.g. """there. \"उनले "": preserve double-quotation before उनले
                    sequence = char + text[index+1]
                    index += 1  # another 1 is added at the end
            else:
                if sequence:
                    sequences.append(sequence)
                    sequence = ''   # Reset sequence
            index += 1

            # print(f'{sequences}\n{sequence}\n{char}{is_devanagari(char)}\n\n')
        if sequence:    # last sequence
            sequences.append(sequence)
        return ' '.join(sequences)
        # Example of using regex for special character removal
    # def remove_non_devanagari_characters(self, text, keep_special_characters=True):
    #     # step-1 : Convert
    #     '''
    #     * Remove all characters that are not(^) part of devanagari characters
    #     * characters that are part of devanagari:


    #     '''

    #     if keep_special_characters:
    #         # Pattern to match continuous Devanagari text (including punctuation) or exclude continuous non-Devanagari segments
    #         pattern = r'[ऀ-ॿ{}]+|[^\sऀ-ॿ]+'.format(re.escape(string.punctuation))
    #         '''
    #         [ऀ-ॿ{}]+    matches sequences of Devanagari characters and punctuation.
    #         [^\sऀ-ॿ]+  matches any continuous sequence of characters that are not whitespace (\s) or Devanagari (ऀ-ॿ)
    #         +          matches one or more occurrences of the preceding element
    #         '''
    #     else:
    #         pattern = r"[^ऀ-ॿ ]"

    #     text = re.sub(pattern, " ", text)

    #     return text
        # Example of using regex for special character removal
    def remove_non_devanagari_characters(self, text, keep_special_characters=True):
        '''
            # Function to find nepali characters. keep punctuations if they occur between devanagari characters. Remove punctuation if previous character is not devanagari.
            # Examples
            texts = [
                "उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। \"hi there\". what is your name? उनले दुहेको दूध",
                "\"hi there. \"उनले दुहेको\" दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। hi there. what is your name? उनले दुहेको दूध\"",
                "name? उनले दुहेको दूध\""    #output: (last quatation, name?) should be ignored
                ]

            for text in texts:
                removed = remove_non_devanagari_characters(text)
                print(f'text: {text}, \nclen: {removed}\n\n')

            
            # output
            text: उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। "hi there". what is your name? उनले दुहेको दूध, 
            clen: उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे।             उनले दुहेको दूध


            text: "hi there. "उनले दुहेको" दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे। hi there. what is your name? उनले दुहेको दूध", 
            clen:    "उनले दुहेको दूध बेच्नका लागि बजार असाध्यै सानो थियो त्यसैले उनले चीज बनाउने विचार गरे।             उनले दुहेको दूध"


            text: name? उनले दुहेको दूध", 
            clen:  उनले दुहेको दूध"
        '''
        def is_devanagari(char):
            pattern=r'[ऀ-ॿ]'
            return bool(re.match(pattern, char))
        sequences = []
        sequence = ''
        punctuation_symbols = string.punctuation    # '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
        prefix_punctuations = '\"\'(<[{'
        index=0
        while index < len(text):
            char = text[index]
            if is_devanagari(char) or char == ' ':
                # Character is devanagari
                sequence += char
            elif char in punctuation_symbols:
                # Character is punctuation
                if sequence != '':
                    if (len(text) > index+1) and not is_devanagari(text[index+1]):
                        # e.g. गरे। "hi there" : skip quotation before hi
                        pass
                    else:
                        sequence += char    # Sequence is no empty. i.e. previous char/sequence was devanagari otherwise ignore  punctuation
                elif (len(text) > index+1) and is_devanagari(text[index+1]):
                    # preserve prefix punctuations in devanagari. e.g. """there. \"उनले "": preserve double-quotation before उनले
                    sequence = char + text[index+1]
                    index += 1  # another 1 is added at the end
            else:
                if sequence:
                    sequences.append(sequence)
                    sequence = ''   # Reset sequence
            index += 1
            
            # print(f'{sequences}\n{sequence}\n{char}{is_devanagari(char)}\n\n')
        if sequence:    # last sequence
            sequences.append(sequence)
        return ' '.join(sequences)
    def normalize_data(self, text):
      '''
        * Standerize special characters
        * e.g. convert different types of quotes to standard quotes
      '''
      characters_to_replace = {
        '₹': 'रु',
        'ʻ': "'",
        'ː': ':',
        '？': 'ॽ',
        '?' : 'ॽ',
        '‟': '"',
        '“' : '"',
        '”': '"',
        '`': "'",
        '৷': '।',
        'ˈ': "'",
        '՛': "'",
        'ǃ': '!',
        '（': '(',
        '：': ':',
        'ˍ': '_',
        '﹣': '-',
        '״': '"',
        'ꞌ': "'",
        '₋': '-',
        '％': '%',
        '꞉': ':',
        '‵': "'"
      }
      # Replace each character in the dictionary with its corresponding standard character
      for char, replacement in characters_to_replace.items():
          text = text.replace(char, replacement)

      return text

    def clean_data(self, text):
        # Remove HTML tags
        text = self.clean_html(text)

        # Normalize some characters
        text = self.normalize_data(text)

        # Convert convert 0-9 to ० - ९
        text = self.convert_to_devanagari_digits(text)

        text = self.remove_non_devanagari_characters(text, keep_special_characters=True)
        # text = text.lower() # No lower characters in devanagari

        # Replace one or more spaces with a single space
        text = re.sub(r'\s+', ' ', text).strip()

        return text



if __name__=="__main__":
    # Clean data and save to 'cleaned_data'
    data_cleaner = CleanData()    
    # Assuming `data` is your raw dataset
    data0 = ["<html><h1>Hi there</h1><style>color:red</style><script>console.log('hello world');</script>Example text</html>", "<div>Another example with special chars: !@#</div>"]
    data = ['<html><h1>सुर्खेत र \n \u200c   </h1><style> जुम्लामा बाहेक <-  this is ignored </style><script>console.log("कर्णालीका अरू जिल्लामा");</script>      कर्णालीका अरू जिल्लामा शिशुका लागि आवश्यक एनआईसीयू सेवा नै उपलब्ध छैन।', 'नेपालले करिब एक महिना अघि नै औपचारिक पत्र पठाएर जीबी राईलाई स्वदेश फर्काइदिन गरेको आग्रहबारे मलेशियाले कुनै औपचारिक जबाफ दिएको छैन।', '2024 बीबीसी। अन्य वेबसाइटका सामग्रीहरूका लागि बीबीसी जिम्मेवार छैन।', 'संसदीय छानबिन समिति गठन नभएसम्म संसद्\u200cको कारबाही अघि बढ्न नदिने प्रतिपक्षी दलको अडानका कारण यसअघिको संसद् अधिवेशनको अन्त्यतिरका कामकारबाही प्रभावित भएका थिए।', 'भारतमा पनि भेपमा प्रतिबन्ध लगाइएको छ तर स्थानीय सञ्चार माध्यमका अनुसार यो व्यवस्था प्रभावकारी रूपमा कार्यान्वयन भएको छैन।', 'इजरेलले रफाहमा आक्रमण गर्न सक्ने आशङ्काका बीच अमेरिकाले गत साता इजरेललाई उपलब्ध गराउन लागेको हजारौँ बमको ढुवानी रोकिदिएको खुलासा भएको छ।', 'शुक्रवारको प्रयत्न सफल भएमा अन्तर्राष्ट्रिय अन्तरिक्ष स्टेशनतिर र बाट उडान गर्ने बोइङ दोस्रो निजी कम्पनी बन्नेछ। अहिलेसम्म इलोन मस्कको स्पेस एक्सले त्यस्तो काम गर्दै आएको छ।', 'हिरादेवी खतिवडाले दैनिक परामर्श दिनेहरूमा जबरजस्ती करणी, यौन दुर्व्यवहार लगायतका घटनाका पीडितहरू हुने गर्छन्।', 'युक्रेनी सेक्युरिटी सर्भिसका अनुसार गिरफ्तार गरिएका दुईजना कर्णेलले जेलेन्स्कीका अङ्गरक्षकहरूमध्ये त्यस्ता व्यक्तिको खोजी गरिरहेका थिए जो उनको अपहरण तथा हत्या गर्नका निम्ति इच्छुक होऊन्।', 'आधिकारिक रूपमा पुटिनले मार्चमा भएको राष्ट्रपतीय चुनावमा ८७ प्रतिशतभन्दा धेरै मत हासिल गरेका थिए। उक्त चुनावमा उनले गम्भीर प्रतिद्वन्दीको सामना गर्नुपरेको थिएन, न उक्त चुनावलाई धेरैले स्वतन्त्र र निष्पक्ष चुनावका रूपमा हेरेका थिए।', 'नेपालका ग्रामीण भेगमा विगतमा पाइने गरेका काठका पुराना ठेकीहरू अचेल नपाइने गरेका मानिसहरू बताउँछन्। आखिर कहाँ जादै छन् सारा ठेकीहरू?', 'चीनको अर्थतन्त्र सोचेभन्दा तीव्र गतिमा विकसित भए पनि यसका अगाडि विभिन्न सङ्कट रहेकाले \nमहिलाहरू घरखर्च धान्न बचत बढाउन चाहन्छन्।', "चार वर्षअघि जारी नेपालको पछिल्लो नक्सालाई अन्तर्राष्ट्रिय मान्यता दिलाउन उल्लेख्य प्रगति नभएको अवस्थामा सरकारको यो निर्णय 'लोकप्रियताका लागि मात्रै गरिएको' कतिपय विज्ञहरूको टिप्पणी छ।", 'जङ्गलमा कुनै प्राणीले औषधीय वनस्पति प्रयोग गरेर चोटपटकको उपचार गरेको यो पहिलो रेकर्ड हो। ', ' ', 'टिकटकले अमेरिकाले प्रतिबन्ध पुष्टि गर्न “अनुमानका आधारमा चिन्ता” व्यक्त गरेको आरोप लगाउँदै उक्त कदम रोक्न अदालतसँग माग गरेको छ।', "पुरातत्त्वसम्बन्धी विशेषज्ञ चित्रकारहरूले सयौँ टुक्राहरू जोडेर बनाइएको खप्परका आधारमा 'नीयान्डर्टाल' महिलाको थ्री-डी मोडल बनाएका छन्। ", 'प्री-मनसुनको समयमा नेपालमा चट्याङ र हावाहुरीका घटनाहरू धेरै हुने गरेको विज्ञहरू बताउँछन्।', 'त्रिसट्ठी वर्षीया चर्चित कलाकारले क्यान्सरविरुद्ध आफ्नो सङ्घर्ष र रङ्गमञ्च एवं सामाजिक सञ्जालमार्फत् आफ्नो यात्राबारे जनचेतना जगाउने निर्णयबारे बीबीसीसँग कुराकानी गरेकी छन्।', 'आस्ट्राजेनेका खोपले कोभिड महामारीका क्रममा लाखौं मनिसहरूको ज्यान बचाएको विश्वास गरिन्छ, तर सँगसँगै यसले केही दुर्लभ रक्तजन्य घातक समस्या पनि निम्त्याएको बताइन्छ। ', 'गत वर्ष अर्थात् सन् २०२३ मा २० हजार जनाभन्दा बढी नेपाली कोरिया गएका थिए। कोरियामा अहिले नेपाली कामदारको सङ्ख्या ५५ हजार जनाभन्दा धेरै रहेको त्यहाँस्थित नेपाली दूतावासले जनाएको छ।', 'एमालेका हिक्मत कार्की मुख्यमन्त्री नियुक्त भएको दिन नै त्यसविरुद्ध मुद्दा लिएर कांग्रेसका केदार कार्की सर्वोच्च अदालत पुगेका छन्।', "महत्त्वाकाङ्क्षी 'नीअम' परियोजनाअन्तर्गत 'द लाइन' मरुभूमि सहरका लागि जग्गा खाली गर्ने काममा घातक बल प्रयोग गर्न साउदी अरबका अधिकारीहरूले अनुमति दिएको एक भूतपूर्व गुप्तचर अधिकारीले दाबी गरेका छन्।\n", 'कडा कार्य संस्कृति प्रोत्साहन गर्ने टिप्पणीलाई लिएर चिनियाँ प्रविधि कम्पनीकी जनसम्पर्क अधिकारीले किन माफी माग्नु पर्\u200dयो।', "केन्द्रमा गत फागुनमा सत्ता समीकरण परिवर्तन भएसँगै त्यसको प्रभाव प्रदेशहरूमा परिरहेको देखिएका बेला जसपा विभाजनले विशेषगरी मधेश प्रदेश सरकार 'ढल्न सक्ने' अनुमानहरू पनि गरिएका छन्।", 'अदालतले उनलाई चुनाव प्रचारप्रसार गर्न कुनै खालको रोक नलगाएको र आदेशमा उनले के गर्न पाउने वा नपाउने भन्ने कुनै विषयबारे उल्लेख नगरिएको पनि केजरीवालका वकिलले बताए।', 'इजरेली प्रधानमन्त्रीले गाजाको रफाहमा पूर्ण स्तरको आक्रमणलाई अनुमति दिएको खण्डमा हतियारको आपूर्ति रोक्न सकिने अमेरिकी चेतावनीलगत्तै बेन्जमिन नेतन्याहुले इजरेल "एक्लै खडा हुन सक्ने" प्रतिक्रिया दिएका छन्।', 'ब्रजिलको रिओ ग्रान्डी डु सुल प्रान्तमा आएको बाढी र पहिरोका कारण धनजनको ठूलो क्षति भएको छ।', 'खासगरी प्रधानमन्त्री, मुख्यमन्त्री एवं प्रदेश प्रमुख परिवर्तन भइरहने अनि मनपरी ढङ्गले आफू अनुकूलका व्यक्तिका तस्बिर राख्दा त्यसको आर्थिकभार राज्यकोषमा पर्ने गरेको जानकारहरू बताउँछन्।', 'नेपाली कांग्रेसले गृहमन्त्री रवि लामिछाने सहकारी ठगीमा संलग्न भएको विवरण सञ्चारमाध्यममा आइरहेको भन्दै उनीमाथि पनि छानबिन हुने गरी संसदीय समिति गठन गरिनुपर्ने अडानमा छ। ', 'हजारौँ आप्रवासीहरूलाई तस्करी गरी यूके पुर्\u200dयाएका स्करपीअन उपनामले चर्चित बार्जान बीबीसीको एउटा अनुसन्धानपछि पक्राउ गरेका छन्। ', 'अमेरिकाका पूर्वराष्ट्रपति डोनल्ड ट्रम्पविरुद्ध परेको मुद्दामा उनका पूर्ववकिल माइकल कोएनले ट्रम्पविरुद्ध बयान दिएका छन्। उनले के भने र उनको बयान किन महत्त्वपूर्ण छ?', 'नेपालमा जेठ १५ गते बजेट ल्याउने नियम छ। त्यसअघि संसद्\u200cमा नीति तथा कार्यक्रम प्रस्तुत हुन्छ। यसपालि मङ्गलवार संसद्को दुवै सदनको संयुक्त बैठक बस्दैछ र त्यसमा राष्ट्रपति रामचन्द्र पौडेलले नीति तथा कार्यक्रम प्रस्तुत गर्ने कार्यक्रम तय भएको छ।', 'उत्तर कोरियाको एउटा प्रान्तमा मानिसहरू भोकभोकै मर्न थालेको सुनेपछि देश छाडेर दक्षिण कोरियामा शरण लिन पुगेका यी व्यक्तिले सन् २०१५ देखि बोतलमा अन्य चिजसँगै चामल भरेर पठाउन थालेका हुन्।\n', "पिनाइल क्यान्सर भनिने पुरुषहरूमा लिङ्गको क्यान्सरका एकजना बिरामी जोआओ भन्छन्, रोगको निदान भएपछि लिङ्ग काटेर हटाउनुपर्ने अवस्थाबाट उनी 'निकै आत्तिएका' थिए।", 'सर्वोच्च अदालतको संवैधानिक इजलासको फैसलाको पूर्णपाठमा हदबन्दीभन्दा बढी भएको जग्गा बाँझो राखिएको भेटिए सरकारका नाममा ल्याउन समेत भनिएको छ।', 'इन्टरनेटमा बालबालिकाले कलिलै उमेरमा पहुँच पाउँदा उनीहरू गम्भीर जोखिमको नजिक पुगेको विशेषज्ञहरूले बताएका छन्। ती जोखिमको रोकथामका लागि तपाईँहामी के गर्न सक्छौँ?', 'मृत्यु सन्निकट हुँदा मानिसहरूले गर्ने अनुभूतिबारे डा क्रिस्टोफर केरले अध्ययन गरेका छन्। त्यस्ता मानिसहरूले अन्तिम समयमा अनुभव गर्ने दृश्य र तिनको अर्थबारे उनले कुरा गरेका छन्।', "आत्मसम्मान कम भएका व्यक्तिमा  'कम्प्लिमेन्ट' अर्थात प्रशंसाले चिन्ता बढाउन सक्छ किनभने उनीहरूको स्वधारणामा त्यसरी भएको तारिफले चुनौती दिन सक्छ। ", 'सन् २०२२ मा जोन म्याकफललाई शारीरिक अपाङ्गता भएका प्रथम अन्तरिक्षयात्रीको उम्मेदवारका रूपमा चयन गरिएको थियो। ', "थाईल्यान्डमा 'लीज म्याजस्टी' कानुनले राजतन्त्रको आलोचना गर्न प्रतिबन्ध लगाएको छ। राजतन्त्रको संरक्षणका लागि बनाइएको त्यस्तो कानुनलाई विश्वकै कठोर कानुनमध्ये मानिन्छ।", "हजारौँ आप्रवासीहरूलाई तस्करी गरी यूके पुर्\u200dयाएका 'स्कोर्पिअन' उपनामले चर्चित बर्जान मजिद बीबीसीको एउटा अनुसन्धानपछि इराकमा पक्राउ परेका छन्। ", 'नयाँ नोट छपाइमा जानुअघि अन्तर्राष्ट्रिय बोलकबोलसहितका कतिपय कानुनी प्रक्रियाहरू पूरा गर्नुपर्ने अधिकारीहरू बताउँछन्। ', 'अमेरिका, चीन र भारतले चन्द्रमामा मानव पठाउने योजना सार्वजनिक गरेका छन्। अन्तरिक्ष अभियानमा यी देशले किन अर्बौँ डलर लगानी गरेका हुन् र उनीहरूको अपेक्षा के छ?', 'जबरजस्ती करणी मुद्दामा उच्च अदालत पाटनले सफाइ दिएको केही घण्टापछि नेपाल क्रिकेट सङ्घ क्यानको निर्णय सार्वजनिक भएको हो।', ' प्रधानमन्त्रीको टाउको र छातीमा गोली लागेको देखिएको थियो। प्रधानमन्त्रीका सुरक्षाकर्मीमध्ये तीन जनाले उनको उद्धार गरी कारभित्र लगेका थिए।', 'मसला उत्पादन र निर्यातमा भारतको वर्चस्व छ। तर विभिन्न देशमा भारतीय मसलाको गुणस्तरबारे प्रश्न उठेको छ।', 'हीरामन्डीमा प्रमुख भूमिका रहेको मल्लिका जानको अभिनय गरेर फिल्म क्षेत्रमा वाहवाही पाइरहेकी मनीषा कोइरालाले बलुवुडकी चर्चित कलाकार रेखाले आफ्नो प्रशंसा गरेको सुन्दा आँशु आएको बताएकी छन्। रेखाले हीरामन्डी हेरेपछि मनीषालाई के भनिन् र उनको आँखा रसाए?', "'लेडी बुशरा' नामले चिनिने अमिर डीन यूकेका सर्वाधिक चर्चित 'ड्र्याग कलाकार'मध्ये एक हुन्।", "साउदी अरबलाई आधुनिक बनाउने युवराज मोहम्मद बिन सलमानको चाहनाअनुसार 'नीअम' परियोजना बनाउन लागिएको हो। \n\n", 'कृषि जनशक्तिको चरम अभावमाझ राष्ट्रपतिले घोषणा गरेको महत्त्वाकाङ्क्षी कृषिमा लगानी दशक कार्यान्वयनमा नआउँदै गम्भीर प्रश्नहरू तेर्सिएका छन्।', 'युक्रेनमा जारी युद्धमा रुसलाई सघाएको आरोप चीनलाई लागेकै बेला भ्लादिमिर पुटिनले फेरि बेइजिङ भ्रमणमा पुगेका छन्।', 'भारतमा निर्वाचनका बेला डीपफेक र एआईबाट सिर्जित भ्रामक सामग्रीको बिगबिगी बढेपछि विज्ञहरूले त्यसबाट पर्ने प्रभावबारे चिन्ता व्यक्त गरेका छन्।', 'जबरजस्ती करणी मुद्दामा उच्च अदालत पाटनले सफाइ दिएको केही घण्टापछि नेपाल क्रिकेट सङ्घ (क्यान) को निर्णय सार्वजनिक भएको हो।', 'उत्तर अमेरिकामा यो पहिलो विश्वकप हुँदै छ भने आयोजकको हैसियतले अमेरिकाले पनि सीधै विश्वकप खेल्ने मौका पाएको छ।', 'सहकारी संस्थाका सञ्चालकहरूले सम्पत्ति ब्याङ्कमा धितो राखेर ऋण लिएको पाइएका कारण लिलामी प्रक्रियामा समस्या भएको अधिकारीहरूले बताएका छन्।', "'ब्रेकिङ'लाई ओलिम्पिक्समा पहिलो पटक समावेश गरिएको हो। यसपालि नै तालिबानको धम्कीका बीच 'ब्रेकिङ' सिकेकी मनिजा तलाश शरणार्थी टोलीकी सदस्यका रूपमा खेल्दै छिन्।"]

    # Preprocess the data
    cleaned_data = [data_cleaner.clean_data(doc) for doc in data]
    print(cleaned_data)

    # Further steps, like tokenization or saving the cleaned data, can follow
