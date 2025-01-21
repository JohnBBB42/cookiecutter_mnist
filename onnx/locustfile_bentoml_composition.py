from locust import HttpUser, between, task

class BentoMLUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def send_prediction_request(self):
        # Open the image file in binary mode for each request
        with open("my_cat.jpg", "rb") as image_file:
            files = {"image_file": ("my_cat.jpg", image_file, "image/jpeg")}
            self.client.post("/predict", files=files)
