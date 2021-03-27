from flask import (render_template, request, redirect, Blueprint,
                   session, url_for, flash, Markup,Request,
                   jsonify,)
import numpy as np
from app.cards import card_data
main = Blueprint('main',__name__)

np.random.seed(41)
np.random.shuffle(card_data)

@main.route("/",methods=["GET","POST"]) 
@main.route("/train",methods=["GET","POST"]) 
def train(): 
    randcard = request.args.get('randcard',0,type=int)
    if randcard:
        print("Choosing random card")
        page_num = np.random.randint(0,len(card_data),1)[0] + 1
    else:
        page_num = request.args.get('page_num',1,type=int)
    card_index = page_num - 1
    card_dict = card_data[card_index]
    n_pages_tot = len(card_data)
    titles = [x['title'] for x in card_data]
    print(titles)
    return render_template('train.html',page_num=page_num,
        n_pages_tot=n_pages_tot,card_dict=card_dict,titles=titles)
