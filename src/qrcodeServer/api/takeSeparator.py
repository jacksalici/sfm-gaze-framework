import qrcode #pypi.python.org/pypi/qrcode
#from pyzbar.pyzbar import decode
from PIL import Image

def create_number(scene, participant, take):
    assert type(scene) == int and type (participant) == int and type(take) == int, "Scene, participant, and take codes must be integers."
    assert scene < 1000 and participant < 1000 and take < 1000, "Codes must be lower than 1000."
    
    number = scene * 1_000_000 + participant * 1000 + take 
    return number

def extract_codes(number):
    assert type(number) == int, "Number must be an integer."
    assert number < 1_000_000_000, "Number must be less than 1,000,000,000."
    
    scene = number // 1_000_000
    participant = (number % 1_000_000) // 1000
    take = number % 1000
    
    return scene, participant, take

def encode(scene, participant, take):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=20, #pixel, in size
        border=4,
    )

    num = create_number(scene, participant, take)

    qr.add_data(num)

    img = qr.make_image(fill_color="black", back_color="white")
    
    return img








