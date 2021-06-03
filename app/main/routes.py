from flask import (render_template, request, redirect, Blueprint,
                   session, url_for, flash, Markup,Request,
                   jsonify,)
import numpy as np
import random


main = Blueprint('main',__name__)

random.seed(0)

@main.route("/",methods=["GET","POST"]) 
@main.route("/train",methods=["GET","POST"]) 
def train(): 
    from app.cards import card_data
    randomize = request.args.get('randomize',0,type=int)
    randseed = request.args.get('randseed',0,type=int)
    page_num = request.args.get('page_num',1,type=int)
    if randomize:
        randseed = np.random.randint(0,100)
    random_card_data = random.Random(randseed).sample(card_data,len(card_data))

    card_index = page_num - 1
    card_dict = random_card_data[card_index]
    n_pages_tot = len(random_card_data)
    titles = [x['title'] for x in random_card_data]
    return render_template('train.html',page_num=page_num,
        randseed=randseed,
        n_pages_tot=n_pages_tot,card_dict=card_dict,titles=titles)
