import math



def number_to_word(num):
    num = str(num)
    # Define units and teens arrays for Nepali words
    # units = ["शुन्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ", "दस"]
    one_to_hundred = ["शुन्य", "एक", "दुई", "तीन", "चार", "पाँच", "छ", "सात", "आठ", "नौ", "दस", "एघार", "बाह्र", "तेह्र", "चौध", "पन्ध्र", "सोह्र", "सत्र", "अठार", "उन्नाइस", "बीस", "एक्काइस", "बाइस", "तेइस", "चौबीस", "पच्चिस", "छब्बीस", "सत्ताइस", "अठ्ठाइस", "उनन्तीस", "तीस", "एकत्तिस", "बत्तिस", "तेत्तिस", "चौँतिस", "पैँतिस", "छत्तिस", "सट्तीस", "अठतीस", "उनन्चालीस", "चालीस", "एकचालीस", "बयालिस", "त्रिचालीस", "चौवालिस", "पैंतालिस", "छयालिस", "सट्चालीस", "अट्चालीस", "उनन्चास", "पचास", "एकाउन्न", "बाउन्न", "त्रिपन्न", "चौवन्न", "पच्पन्न", "छपन्न", "सन्ताउन्न", "अन्ठाउँन्न", "उनान्न्साठी ", "साठी", "एकसट्ठी", "बैसट्ठी", "त्रिसट्ठी", "चौंसट्ठी", "पैंसट्ठी", "छैसट्ठी", "सतसट्ठी", "अठसट्ठी", "उनन्सत्तरी", "सत्तरी", "एकहत्तर", "बहत्तर", "त्रिहत्तर", "चौहत्तर", "पचहत्तर", "छहत्तर", "सतहत्तर", "अठ्हत्तर", "उनास्सी", "अस्सी", "एकासी", "बयासी", "त्रीयासी", "चौरासी", "पचासी", "छयासी", "सतासी", "अठासी", "उनान्नब्बे", "नब्बे", "एकान्नब्बे", "बयान्नब्बे", "त्रियान्नब्बे", "चौरान्नब्बे", "पंचान्नब्बे", "छयान्नब्बे", "सन्तान्‍नब्बे", "अन्ठान्नब्बे", "उनान्सय"]
    place_values = ['सय', 'हजार', 'लाख', 'करोड', 'अर्ब', 'अर्ब', 'नील', 'पद्म', 'शंख']


    '''
    step 1: Get different placeholders. seperate by : 3,2,2,2... from right to left
    e.g. 1234567 -> 12,34,567

    number of seperator
    len(a) <= 3 : 0   e.g. 123
    len(a) > 3 < 5 : 1 e.g. 12,345
    len(a) > 5 < 7 : 2 e.g. 34,12,345

    i.e. num_sepeartors = 0 if len(a) < 3 else math.floor((len(a)-2)//2)
    note: math.floor can be replaced by // operator
    '''

    num_sepeartors = 0 if len(num) < 3 else math.ceil((len(num)-2)//2)

    seperated_digits = []
    for i in range(num_sepeartors+1):
        if i == 0:
            # Seperate Three digits at the end
            seperated_digits.append(num[-3:])
            num=num[:-3]
        else:
            # Seperate Two digits from start
            seperated_digits.append(num[-2:])
            num=num[:-2]
    # Reverse the list
    seperated_digits = seperated_digits[::-1]
    print(f'seperated_digits:{seperated_digits}')    # ['1', '23', '456']

    '''
    step 2: Get the value for each seperated word
    e.g. ['1', '23', '456'] -> ['एक', 'तेईस', 'चारसट्ठी']


    step 3: Add place value to each seperated word
    e.g. ['एक', 'तेईस', 'चारसट्ठी'] -> ['एक हजार', 'तेईस हजार', 'चारसट्ठी']
    '''

    seperated_values = []
    for i, val in enumerate(seperated_digits):
        if len(str(val)) < 3:
            seperated_values.append(one_to_hundred[int(val)])
            # Add place value
            seperated_values.append(place_values[num_sepeartors - i])
        else:
            '''
            e.g. 123 -> 1, 23
            '''
            if val[0] == '0':
                # e.g. 012 -> 12
                seperated_values.append(one_to_hundred[int(val[1:])])

                
            else:
                # e.g. 123 -> 1 <hundred>, 23 <one_to_hundred>
                seperated_values.extend([
                    one_to_hundred[int(val[0])],
                    'सय',
                    one_to_hundred[int(val[1:])]])
    # print(f'seperated_values: {seperated_values}')  # ['एक', 'लाख', 'तेइस', 'हजार', 'चार', 'सय', 'छपन्न']
    return seperated_values

def generate(n):
    return [number_to_word(i) for i in range(n)]

if __name__ == "__main__":
    for i in range(10000):
        print(f'{i}: {number_to_word(i)}')
