from flask import Blueprint, g, redirect, render_template, request, url_for, session,Response, current_app
from .sysconfig import get_sysconfig
from .component import face_utils
import os



bp = Blueprint('mainflow', __name__, url_prefix='/main-flow')

@bp.route('/delete', methods=["post"])
def delete():
    name = request.form['name']
    face_utils.delete_face_folder(get_sysconfig(), name)

    return redirect('/')

@bp.route('/step01', methods=["POST"])
def step01():
    name = request.form['name']
    data = {'name':name}
    return render_template('step_01.html', data=data)

@bp.route('/step02', methods=["POST"])
def step02():
    name = request.form['name']
    data = {'name':name}
    return render_template('step_02.html', data=data)

@bp.route('/step03/<name>')
def step03(name):
    data = {'name':name}
    return render_template('step_03.html', data=data)