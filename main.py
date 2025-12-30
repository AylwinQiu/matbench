import streamlit as sl
state = sl.session_state
from multiprocessing import Process
state.run_part = ['main']
if "first_run" not in state:
    # The code just run the first time here.
    state.in_add_task = False
    state.pages = {
        "Console":[],
        "Tasks list":[],
    }

def render_main():
    sl.title("Matbench")
    if sl.button("Add task") or state.in_add_task:
        state.in_add_task = True
        task = sl.text_input("1. What type of task you want to add")
        scripts = __import__("mylib.scripts").scripts
        if task in dir(scripts):
            state.pages["Tasks list"].append(sl.Page(getattr(scripts, task), title=task))
            state.in_add_task = False
            sl.rerun()

if "first_run" not in state:
    state.pages["Console"].append(sl.Page(render_main, title="main", icon="👌"))

print(state.pages["Tasks list"])
sl.navigation(state.pages).run()

# This code must be the last part of this script.
if "first_run" not in state:
    state.first_run = False