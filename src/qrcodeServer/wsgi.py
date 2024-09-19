from flask import Flask, send_file, redirect, render_template
from takeSeparator import encode
import io
import base64


app = Flask(__name__)

@app.errorhandler(404)
def error404(error):
    return redirect("/0/0/0")

@app.route("/<scene>/<participant>/<take>")
def qrcode_generator(scene, participant, take):
    qrcode_image = encode(int(scene), int(participant), int(take))
    file = io.BytesIO()
    qrcode_image.save(file, "PNG")
    file.seek(0)
    base64_image = base64.b64encode(file.getvalue()).decode('utf-8')

    return render_template('std.html', image = base64_image, scene = scene, participant = participant, take = take)
    