import numpy as np
from flask import Flask, abort, jsonify, request
import pickle
import os

my_model = pickle.load(open('airbnb_model.pkl', 'rb'))

app = Flask(__name__)


@app.route('/api', methods=['POST'])
def make_predict():
    # get data from post (4 features)
    data = request.get_json(force=True)
    # # transforms

    # #
    # # change the columns ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # #
    # predict_request = [data['room_type'], data['number_of_reviews'], data['calculated_host_listings_count'], \
    #     data['availability_365'],data['distance'], data['cancellation_policy'], data['size'], data['amenities_num'], \
    #         data['host_identity_verified'], data['security_deposit'], data['cleaning_fee'], \
    #              data['guests_included'],data['extra_people'], data['review_scores_rating'], data['bathrooms'], \
    #                  data['bedrooms'], data['beds'], data['bed_type'], data['accommodates']]
    # predict_request = np.array(predict_request).reshape(1, -1)
    # # make predictions
    # y_hat = my_model.predict(predict_request)
    # # send preds back
    # output = {'y_hat': int(y_hat[0])}
    # return jsonify(results=output)
    print("H E R E")
    print(data)
    return jsonify(results=data)

if __name__ == '__main__':
    app.run(port=9000, debug=True)
