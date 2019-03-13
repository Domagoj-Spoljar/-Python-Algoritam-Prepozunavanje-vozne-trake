
line_length=84

def print_line_in_defined_length(text,length):
    text_length=len(text)
    additional_length=length-text_length
    out_text='|'+text+' '*additional_length+'|'
    return out_text

def print_line_text_in_middle(text,length):
    text_length=len(text)
    additional_length=length-text_length
    if additional_length%2:
        first_half=additional_length//2
        second_half=first_half+1
    else:
        first_half=additional_length//2
        second_half=additional_length//2

    out_text='|'+' '*first_half+text+' '*second_half+'|'
    return out_text

def print_line_3_columns(text1,text2,text3,length):
    text_length1=len(text1)
    text_length2=len(text2)
    text_length3=len(text3)
    col_length=length//3
    additional_length1=col_length-text_length1
    additional_length2=col_length-text_length2
    additional_length3=col_length-text_length3
    first_half1=0
    second_half1=0
    first_half2=0
    second_half2=0
    first_half3=0
    second_half3=0

    if additional_length1%2:
        first_half1=additional_length1//2
        second_half1=first_half1+1
    else:
        first_half1=additional_length1//2
        second_half1=additional_length1//2

    if additional_length2%2:
        first_half2=additional_length2//2
        second_half2=first_half2+1
    else:
        first_half2=additional_length2//2
        second_half2=additional_length2//2

    if additional_length3%2:
        first_half3=additional_length3//2
        second_half3=first_half3
    else:
        first_half3=additional_length3//2
        second_half3=additional_length3//2-1

    out_text='|'+' '*first_half1+text1+' '*second_half1+'|' +' '*first_half2+text2+' '*second_half2+'|' +' '*first_half3+text3+' '*second_half3+'|'
    return out_text
