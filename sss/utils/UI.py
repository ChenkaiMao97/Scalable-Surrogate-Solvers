def printc(text, color=None):
    if color == 'r':
        print("\033[91m" + text + "\033[0m")
    elif color == 'g':
        print("\033[92m" + text + "\033[0m")
    elif color == 'y':
        print("\033[93m" + text + "\033[0m")
    elif color == 'b':
        print("\033[94m" + text + "\033[0m")
    else:
        print(text)
