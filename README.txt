# Facematch

Seeing how facecheck.id is no longer working (you need to pay), I figured the best other option is to compare images between sites for biometric purposes.

***Requirenents***

* A GPU
* Python 3+ / pip
* cmake
* 
# Install

	python3.10 -m venv venv /
	source venv/bin/activate /
	pip3 install git+https://github.com/ageitgey/face_recognition_models /
	pip3 install --upgrade pip setuptools wheel /

to run simply edit the python file lines with the found images:


    target = "https://imageio.forbes.com/specials-images/imageserve/66f5b8cbf6d5e9f3f3703478/1x1-Gabe-Newell-credit-Edge-Magazine-getty-images/0x0.jpg?format=jpg&height=1080&width=1080"
    candidates = [
    "https://static.wikia.nocookie.net/half-life/images/6/62/Gaben.jpg/revision/latest/scale-to-width-down/1200?cb=20200126040848&path-prefix=en",
    "https://sm.ign.com/ign_nordic/news/v/valve-boss/valve-boss-gabe-newell-says-hes-been-retired-in-a-sense-for_n2vz.jpg"
    ]

Then run hash.py >> hashes.txt
