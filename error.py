

def _parse_msg(msg):
    return msg.replace("\n", " ")


def qerror(e, msg):
    return e(msg)
    