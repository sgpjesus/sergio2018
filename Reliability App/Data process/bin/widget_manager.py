"""
This module has purpose of managing the usage of widgets in the jupiter
notebooks of the reliability application, reducing the code in the notebooks,
and making everything readable.
"""

import ipywidgets as widgets

from IPython.display import display


class Display:
    """
    Generic display of a jupyter notebook
    """

    def __init__(self):
        """
        Initialization of the display.
        """
        self._existing_widgets = []

    def create_widget(self, widget_name, **widget_params):
        temporary_widget = getattr(widgets, widget_name)()
        for key, value in widget_params.items():
            setattr(temporary_widget, key, value)
        self._existing_widgets.append(temporary_widget)

    def display_widgets(self):
        for widget in self._existing_widgets:
            display(widget)

    def delete_widgets(self, widget_index=-1):
        del(self._existing_widgets[widget_index])
