import qrcode

qr = qrcode.QRCode(
    version=1,
    error_correction=qrcode.constants.ERROR_CORRECT_H,
    box_size=10, #pixel, in size
    border=4,
)

qr.clear()

qr.add_data(b'GD_01_02_03')

img = qr.make_image(fill_color="red", back_color="white")

img.save("some_file1.png")