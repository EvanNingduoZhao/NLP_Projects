def pal(wrd):
    wrd_no_space = wrd.rstrip()
    rvs_wrd = ''
    for i in wrd:
        rvs_wrd= i + rvs_wrd
    return "This is a palindrome" if rvs_wrd == wrd_no_space else "This is not a palindrome"

# test
print("pal('racecar'): " + str(pal('racecar')))
print("pal('palindrome'): " + str(pal('palindrome')))
print("pal('a dog a plan a canal pagoda'): " + str(pal('a dog a plan a canal pagoda')))  fvd