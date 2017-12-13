import traceback


OPACITY = 0.4
remote_API = False
BLUE = (232, 167, 35)
BRAND = "partypi.net"
VERSION = "0.1.5"
PURPLE = (68, 54, 66)
hat_path = 'images/hat.png'


def print_traceback():
    print ("Exception:")
    print ('-' * 60)
    traceback.print_exc()
    print ('-' * 60)
    pass
