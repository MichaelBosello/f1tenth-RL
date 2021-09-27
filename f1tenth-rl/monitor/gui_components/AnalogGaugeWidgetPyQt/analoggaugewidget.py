#!/usr/bin/env python

###
# Author: Stefan Holstein
# inspired by: https://github.com/Werkov/PyQt4/blob/master/examples/widgets/analogclock.py
# Thanks to https://stackoverflow.com/
#
# Sorry for mixing english & german notes
#
# ToDo: Fix Bug: Rundungsfehler Max Value / Grid
# ToDo: mehrere Zeiger ermöglichen. z.b. über ein ZeigerArray mit allen valiablen
#       Signal erzeugung (self.valueChange.emit()) pruefen wie es dann möglich ist.
#       Evtl MausTracking(Teil)-deaktivieren
#       Farben separat handeln
# todo: aktuell ist nur eine Zeigerrichtung klein nach gross im Uhrzeigersinn moeglich
# -> erweiterung Anzeige von gross nach klein um Uhrzeigersin
# todo: auf timer event verzichten um effizienz zu steigern
#       self.update() an allen stellen einfügen, an denen es notwendig ist.
#       It is possible to En-/disable timerevents. Use: self.use_timer_event = True/False
# todo: Bug Fix: Offset Berechnung bezogen auf den Winkel ist falsch
# Todo: print() in logging() ausgabe aendern
###

import math

try:
    # print("trying to import Qt4 @ analoggaugewidget.py")
    from PyQt4.QtGui import QMainWindow

    from PyQt4.QtGui import QWidget
    from PyQt4.QtGui import QApplication
    from PyQt4.QtGui import QPolygon, QPolygonF, QColor, QPen, QFont
    from PyQt4.QtGui import QPainter, QFontMetrics, QConicalGradient
    # QtGui -> QPolygon, QPolygonF, QColor, QPen, QFont,
    #       -> QWidget
    #       -> QApplication

    from PyQt4.QtCore import Qt, QTime, QTimer, QPoint, QPointF, SIGNAL, QRect, QSize
    from PyQt4.QtCore import QObject, pyqtSignal
    # QtCore -> Qt.NoPen, QTime, QTimer, QPoint, QPointF, QRect, QSize


    used_Qt_Version = 4
    print("end trying to import Qt4 @ analoggaugewidget.py")
    # Antialysing may be problem with Qt4
    print("ToDo: Fix error output QPainter.Antialiasing")

except:
    try:
        # print("Try5: analoggaugewidget.py")
        from PyQt5.QtWidgets import QMainWindow

        from PyQt5.QtWidgets import QWidget
        from PyQt5.QtWidgets import QApplication
        # QtWidgets -> QWidget
        # QtWidgets -> QApplication

        from PyQt5.QtGui import QPolygon, QPolygonF, QColor, QPen, QFont
        from PyQt5.QtGui import QPainter, QFontMetrics, QConicalGradient
        # QtGui -> QPolygon, QPolygonF, QColor, QPen, QFont, QPainter, QFontMetrics, QConicalGradient

        from PyQt5.QtCore import Qt ,QTime, QTimer, QPoint, QPointF, QRect, QSize
        from PyQt5.QtCore import QObject, pyqtSignal
        # QtCore -> Qt.NoPen ,QTime, QTimer, QPoint, QPointF, QRect, QSize

        used_Qt_Version = 5
        print("end trying to import Qt5 @ analoggaugewidget.py")
    except:
        print("Error Import Qt 4 & 5 @ analoggaugewidget.py")
        exit()

##########################################
# todo: Dokumentieren
##########################################

class AnalogGaugeWidget(QWidget):
    """Fetches rows from a Bigtable.
    Args: 
        none
    
    """
    valueChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super(AnalogGaugeWidget, self).__init__(parent)

        self.use_timer_event = False
        self.black = QColor(0, 0, 0, 255)

        # self.valueColor = QColor(50, 50, 50, 255)
        # self.set_valueColor(50, 50, 50, 255)
        # self.NeedleColor = QColor(50, 50, 50, 255)
        self.set_NeedleColor(50, 50, 50, 255)
        self.NeedleColorReleased = self.NeedleColor
        # self.NeedleColorDrag = QColor(255, 0, 00, 255)
        self.set_NeedleColorDrag(255, 0, 00, 255)

        self.set_ScaleValueColor(50, 50, 50, 255)
        self.set_DisplayValueColor(50, 50, 50, 255)

        # self.CenterPointColor = QColor(50, 50, 50, 255)
        self.set_CenterPointColor(50, 50, 50, 255)

        # self.valueColor = black
        # self.black = QColor(0, 0, 0, 255)

        self.value_needle_count = 1
        self.value_needle = QObject
        self.change_value_needle_style([QPolygon([
            QPoint(4, 4),
            QPoint(-4, 4),
            QPoint(-3, -120),
            QPoint(0, -126),
            QPoint(3, -120)
        ])])

        self.value_min = 0
        self.value_max = 1000
        self.value = self.value_min
        self.value_offset = 0
        self.value_needle_snapzone = 0.05
        self.last_value = 0

        # self.value2 = 0
        # self.value2Color = QColor(0, 0, 0, 255)

        self.gauge_color_outer_radius_factor = 1
        self.gauge_color_inner_radius_factor = 0.95
        self.center_horizontal_value = 0
        self.center_vertical_value = 0
        self.debug1 = None
        self.debug2 = None
        self.scale_angle_start_value = 135
        self.scale_angle_size = 270
        self.angle_offset = 0

        # self.scala_main_count = 10
        self.set_scala_main_count(10)
        self.scala_subdiv_count = 5

        self.pen = QPen(QColor(0, 0, 0))
        self.font = QFont('Decorative', 20)

        self.scale_polygon_colors = []
        self.set_scale_polygon_colors([[.00, Qt.red],
                                     [.1, Qt.yellow],
                                     [.15, Qt.green],
                                     [1, Qt.transparent]])

        # initialize Scale value text
        # self.enable_scale_text = True
        self.set_enable_ScaleText(True)
        self.scale_fontname = "Decorative"
        self.initial_scale_fontsize = 15
        self.scale_fontsize = self.initial_scale_fontsize

        # initialize Main value text
        self.enable_value_text = True
        self.value_fontname = "Decorative"
        self.initial_value_fontsize = 40
        self.value_fontsize = self.initial_value_fontsize
        self.text_radius_factor = 0.7

        # En/disable scale / fill
        # self.enable_barGraph = True
        self.set_enable_barGraph(True)
        # self.enable_filled_Polygon = True
        self.set_enable_filled_Polygon(True)


        self.enable_CenterPoint = True
        self.enable_fine_scaled_marker = True
        self.enable_big_scaled_marker = True

        self.needle_scale_factor = 0.8
        self.enable_Needle_Polygon = True

        # necessary for resize
        self.setMouseTracking(False)

        # QTimer sorgt für neu Darstellung alle X ms
        # evtl performance hier verbessern mit self.update() und self.use_timer_event = False
        # todo: self.update als default ohne ueberpruefung, ob self.use_timer_event gesetzt ist oder nicht
        # Timer startet alle 10ms das event paintEvent
        if self.use_timer_event:
            timer = QTimer(self)
            timer.timeout.connect(self.update)
            timer.start(10)
        else:
            self.update()

        self.setWindowTitle("Analog Gauge")

        # self.connect(self, SIGNAL("resize()"), self.rescaleMethod)

        # self.resize(300 , 300)
        self.rescale_method()



    def rescale_method(self):
        # print("slotMethod")
        if self.width() <= self.height():
            self.widget_diameter = self.width()
        else:
            self.widget_diameter = self.height()

        self.change_value_needle_style([QPolygon([
            QPoint(4, 30),
            QPoint(-4, 30),
            QPoint(-2, - self.widget_diameter / 2 * self.needle_scale_factor),
            QPoint(0, - self.widget_diameter / 2 * self.needle_scale_factor - 6),
            QPoint(2, - self.widget_diameter / 2 * self.needle_scale_factor)
        ])])
        # needle = [QPolygon([
        #     QPoint(4, 4),
        #     QPoint(-4, 4),
        #     QPoint(-3, -120),
        #     QPoint(0, -126),
        #     QPoint(3, -120)])]
        # print(str(type(needle)).split("'")[1])
        #
        # needle = [2]
        # print(str(type(needle[0])).split("'")[1])

        self.scale_fontsize = self.initial_scale_fontsize * self.widget_diameter / 400
        self.value_fontsize = self.initial_value_fontsize * self.widget_diameter / 400

        # print("slotMethod end")
        pass

    def change_value_needle_style(self, design):
        # prepared for multiple needle instrument
        self.value_needle = []
        for i in design:
            self.value_needle.append(i)
        if not self.use_timer_event:
            self.update()

    def update_value(self, value, mouse_controlled = False):
        # if not mouse_controlled:
        #     self.value = value
        #
        # if mouse_controlled:
        #     self.valueChanged.emit(int(value))

        if value <= self.value_min:
            self.value = self.value_min
        elif value >= self.value_max:
            self.value = self.value_max
        else:
            self.value = value
        # self.paintEvent("")
        self.valueChanged.emit(int(value))
        # print(self.value)

        # ohne timer: aktiviere self.update()
        if not self.use_timer_event:
            self.update()

    def update_angle_offset(self, offset):
        self.angle_offset = offset
        if not self.use_timer_event:
            self.update()

    def center_horizontal(self, value):
        self.center_horizontal_value = value
        # print("horizontal: " + str(self.center_horizontal_value))

    def center_vertical(self, value):
        self.center_vertical_value = value
        # print("vertical: " + str(self.center_vertical_value))

    ###############################################################################################
    # Set Methods
    ###############################################################################################
    def set_NeedleColor(self, R=50, G=50, B=50, Transparency=255):
        # Red: R = 0 - 255
        # Green: G = 0 - 255
        # Blue: B = 0 - 255
        # Transparency = 0 - 255
        self.NeedleColor = QColor(R, G, B, Transparency)
        self.NeedleColorReleased = self.NeedleColor

        if not self.use_timer_event:
            self.update()

    def set_NeedleColorDrag(self, R=50, G=50, B=50, Transparency=255):
        # Red: R = 0 - 255
        # Green: G = 0 - 255
        # Blue: B = 0 - 255
        # Transparency = 0 - 255
        self.NeedleColorDrag = QColor(R, G, B, Transparency)

        if not self.use_timer_event:
            self.update()

    def set_ScaleValueColor(self, R=50, G=50, B=50, Transparency=255):
        # Red: R = 0 - 255
        # Green: G = 0 - 255
        # Blue: B = 0 - 255
        # Transparency = 0 - 255
        self.ScaleValueColor = QColor(R, G, B, Transparency)

        if not self.use_timer_event:
            self.update()

    def set_DisplayValueColor(self, R=50, G=50, B=50, Transparency=255):
        # Red: R = 0 - 255
        # Green: G = 0 - 255
        # Blue: B = 0 - 255
        # Transparency = 0 - 255
        self.DisplayValueColor = QColor(R, G, B, Transparency)

        if not self.use_timer_event:
            self.update()

    def set_CenterPointColor(self, R=50, G=50, B=50, Transparency=255):
        self.CenterPointColor = QColor(R, G, B, Transparency)

        if not self.use_timer_event:
            self.update()

    def set_enable_Needle_Polygon(self, enable = True):
        self.enable_Needle_Polygon = enable

        if not self.use_timer_event:
            self.update()

    def set_enable_ScaleText(self, enable = True):
        self.enable_scale_text = enable

        if not self.use_timer_event:
            self.update()


    def set_enable_barGraph(self, enable = True):
        self.enable_barGraph = enable

        if not self.use_timer_event:
            self.update()

    def set_enable_value_text(self, enable = True):
        self.enable_value_text = enable

        if not self.use_timer_event:
            self.update()

    def set_enable_CenterPoint(self, enable = True):
        self.enable_CenterPoint = enable

        if not self.use_timer_event:
            self.update()

    def set_enable_filled_Polygon(self, enable = True):
        self.enable_filled_Polygon = enable

        if not self.use_timer_event:
            self.update()

    def set_enable_big_scaled_grid(self, enable = True):
        self.enable_big_scaled_marker = enable

        if not self.use_timer_event:
            self.update()

    def set_enable_fine_scaled_marker(self, enable = True):
        self.enable_fine_scaled_marker = enable

        if not self.use_timer_event:
            self.update()

    def set_scala_main_count(self, count):
        if count < 1:
            count = 1
        self.scala_main_count = count

        if not self.use_timer_event:
            self.update()

    def set_MinValue(self, min):
        if self.value < min:
            self.value = min
        if min >= self.value_max:
            self.value_min = self.value_max - 1
        else:
            self.value_min = min

        if not self.use_timer_event:
            self.update()

    def set_MaxValue(self, max):
        if self.value > max:
            self.value = max
        if max <= self.value_min:
            self.value_max = self.value_min + 1
        else:
            self.value_max = max

        if not self.use_timer_event:
            self.update()

    def set_start_scale_angle(self, value):
        # Value range in DEG: 0 - 360
        self.scale_angle_start_value = value
        # print("startFill: " + str(self.scale_angle_start_value))

        if not self.use_timer_event:
            self.update()

    def set_total_scale_angle_size(self, value):
        self.scale_angle_size = value
        # print("stopFill: " + str(self.scale_angle_size))

        if not self.use_timer_event:
            self.update()

    def set_gauge_color_outer_radius_factor(self, value):
        self.gauge_color_outer_radius_factor = float(value) / 1000
        # print(self.gauge_color_outer_radius_factor)

        if not self.use_timer_event:
            self.update()

    def set_gauge_color_inner_radius_factor(self, value):
        self.gauge_color_inner_radius_factor = float(value) / 1000
        # print(self.gauge_color_inner_radius_factor)

        if not self.use_timer_event:
            self.update()

    def set_scale_polygon_colors(self, color_array):
        # print(type(color_array))
        if 'list' in str(type(color_array)):
            self.scale_polygon_colors = color_array
        elif color_array == None:
            self.scale_polygon_colors = [[.0, Qt.transparent]]
        else:
            self.scale_polygon_colors = [[.0, Qt.transparent]]

        if not self.use_timer_event:
            self.update()

    ###############################################################################################
    # Get Methods
    ###############################################################################################

    def get_value_max(self):
        return self.value_max

    ###############################################################################################
    # Painter
    ###############################################################################################

    def create_polygon_pie(self, outer_radius, inner_raduis, start, lenght):
        polygon_pie = QPolygonF()
        # start = self.scale_angle_start_value
        # start = 0
        # lenght = self.scale_angle_size
        # lenght = 180
        # inner_raduis = self.width()/4
        # print(start)
        n = 360     # angle steps size for full circle
        # changing n value will causes drawing issues
        w = 360 / n   # angle per step
        # create outer circle line from "start"-angle to "start + lenght"-angle
        x = 0
        y = 0

        # todo enable/disable bar graf here
        if not self.enable_barGraph:
            # float_value = ((lenght / (self.value_max - self.value_min)) * (self.value - self.value_min))
            lenght = int(round((lenght / (self.value_max - self.value_min)) * (self.value - self.value_min)))
            # print("f: %s, l: %s" %(float_value, lenght))
            pass

        # mymax = 0

        for i in range(lenght+1):                                              # add the points of polygon
            t = w * i + start - self.angle_offset
            x = outer_radius * math.cos(math.radians(t))
            y = outer_radius * math.sin(math.radians(t))
            polygon_pie.append(QPointF(x, y))
        # create inner circle line from "start + lenght"-angle to "start"-angle
        for i in range(lenght+1):                                              # add the points of polygon
            # print("2 " + str(i))
            t = w * (lenght - i) + start - self.angle_offset
            x = inner_raduis * math.cos(math.radians(t))
            y = inner_raduis * math.sin(math.radians(t))
            polygon_pie.append(QPointF(x, y))

        # close outer line
        polygon_pie.append(QPointF(x, y))
        return polygon_pie

    def draw_filled_polygon(self, outline_pen_with=0):
        if not self.scale_polygon_colors == None:
            painter_filled_polygon = QPainter(self)
            painter_filled_polygon.setRenderHint(QPainter.Antialiasing)
            # Koordinatenursprung in die Mitte der Flaeche legen
            painter_filled_polygon.translate(self.width() / 2, self.height() / 2)

            painter_filled_polygon.setPen(Qt.NoPen)

            self.pen.setWidth(outline_pen_with)
            if outline_pen_with > 0:
                painter_filled_polygon.setPen(self.pen)

            colored_scale_polygon = self.create_polygon_pie(
                ((self.widget_diameter / 2) - (self.pen.width() / 2)) * self.gauge_color_outer_radius_factor,
                (((self.widget_diameter / 2) - (self.pen.width() / 2)) * self.gauge_color_inner_radius_factor),
                self.scale_angle_start_value, self.scale_angle_size)

            gauge_rect = QRect(QPoint(0, 0), QSize(self.widget_diameter / 2 - 1, self.widget_diameter - 1))
            grad = QConicalGradient(QPointF(0, 0), - self.scale_angle_size - self.scale_angle_start_value +
                                    self.angle_offset - 1)

            # todo definition scale color as array here
            for eachcolor in self.scale_polygon_colors:
                grad.setColorAt(eachcolor[0], eachcolor[1])
            # grad.setColorAt(.00, Qt.red)
            # grad.setColorAt(.1, Qt.yellow)
            # grad.setColorAt(.15, Qt.green)
            # grad.setColorAt(1, Qt.transparent)
            painter_filled_polygon.setBrush(grad)
            # self.brush = QBrush(QColor(255, 0, 255, 255))
            # painter_filled_polygon.setBrush(self.brush)
            painter_filled_polygon.drawPolygon(colored_scale_polygon)
            # return painter_filled_polygon

    ###############################################################################################
    # Scale Marker
    ###############################################################################################

    def draw_big_scaled_markter(self):
        my_painter = QPainter(self)
        my_painter.setRenderHint(QPainter.Antialiasing)
        # Koordinatenursprung in die Mitte der Flaeche legen
        my_painter.translate(self.width() / 2, self.height() / 2)

        # my_painter.setPen(Qt.NoPen)
        self.pen = QPen(QColor(0, 0, 0, 255))
        self.pen.setWidth(2)
        # # if outline_pen_with > 0:
        my_painter.setPen(self.pen)

        my_painter.rotate(self.scale_angle_start_value - self.angle_offset)
        steps_size = (float(self.scale_angle_size) / float(self.scala_main_count))
        scale_line_outer_start = self.widget_diameter/2
        scale_line_lenght = (self.widget_diameter / 2) - (self.widget_diameter / 20)
        # print(stepszize)
        for i in range(self.scala_main_count+1):
            my_painter.drawLine(scale_line_lenght, 0, scale_line_outer_start, 0)
            my_painter.rotate(steps_size)

    def create_scale_marker_values_text(self):
        painter = QPainter(self)
        # painter.setRenderHint(QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.Antialiasing)

        # Koordinatenursprung in die Mitte der Flaeche legen
        painter.translate(self.width() / 2, self.height() / 2)
        # painter.save()
        font = QFont(self.scale_fontname, self.scale_fontsize)
        fm = QFontMetrics(font)

        pen_shadow = QPen()

        pen_shadow.setBrush(self.ScaleValueColor)
        painter.setPen(pen_shadow)

        text_radius_factor = 0.8
        text_radius = self.widget_diameter/2 * text_radius_factor

        scale_per_div = int((self.value_max - self.value_min) / self.scala_main_count)

        angle_distance = (float(self.scale_angle_size) / float(self.scala_main_count))
        for i in range(self.scala_main_count + 1):
            # text = str(int((self.value_max - self.value_min) / self.scala_main_count * i))
            text = str(int(self.value_min + scale_per_div * i))
            w = fm.width(text) + 1
            h = fm.height()
            painter.setFont(QFont(self.scale_fontname, self.scale_fontsize))
            angle = angle_distance * i + float(self.scale_angle_start_value - self.angle_offset)
            x = text_radius * math.cos(math.radians(angle))
            y = text_radius * math.sin(math.radians(angle))
            # print(w, h, x, y, text)
            text = [x - int(w/2), y - int(h/2), int(w), int(h), Qt.AlignCenter, text]
            painter.drawText(text[0], text[1], text[2], text[3], text[4], text[5])
        # painter.restore()

    def create_fine_scaled_marker(self):
        #  Description_dict = 0
        my_painter = QPainter(self)

        my_painter.setRenderHint(QPainter.Antialiasing)
        # Koordinatenursprung in die Mitte der Flaeche legen
        my_painter.translate(self.width() / 2, self.height() / 2)

        my_painter.setPen(Qt.black)
        my_painter.rotate(self.scale_angle_start_value - self.angle_offset)
        steps_size = (float(self.scale_angle_size) / float(self.scala_main_count * self.scala_subdiv_count))
        scale_line_outer_start = self.widget_diameter/2
        scale_line_lenght = (self.widget_diameter / 2) - (self.widget_diameter / 40)
        for i in range((self.scala_main_count * self.scala_subdiv_count)+1):
            my_painter.drawLine(scale_line_lenght, 0, scale_line_outer_start, 0)
            my_painter.rotate(steps_size)

    def create_values_text(self):
        painter = QPainter(self)
        # painter.setRenderHint(QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.Antialiasing)

        # Koordinatenursprung in die Mitte der Flaeche legen
        painter.translate(self.width() / 2, self.height() / 2)
        # painter.save()
        # xShadow = 3.0
        # yShadow = 3.0
        font = QFont(self.value_fontname, self.value_fontsize)
        fm = QFontMetrics(font)

        pen_shadow = QPen()

        pen_shadow.setBrush(self.DisplayValueColor)
        painter.setPen(pen_shadow)

        text_radius = self.widget_diameter / 2 * self.text_radius_factor

        # angle_distance = (float(self.scale_angle_size) / float(self.scala_main_count))
        # for i in range(self.scala_main_count + 1):
        text = str(int(self.value))
        w = fm.width(text) + 1
        h = fm.height()
        painter.setFont(QFont(self.value_fontname, self.value_fontsize))

        # Mitte zwischen Skalenstart und Skalenende:
        # Skalenende = Skalenanfang - 360 + Skalenlaenge
        # Skalenmitte = (Skalenende - Skalenanfang) / 2 + Skalenanfang
        angle_end = float(self.scale_angle_start_value + self.scale_angle_size - 360)
        angle = (angle_end - self.scale_angle_start_value) / 2 + self.scale_angle_start_value

        x = text_radius * math.cos(math.radians(angle))
        y = text_radius * math.sin(math.radians(angle))
        # print(w, h, x, y, text)
        text = [x - int(w/2), y - int(h/2), int(w), int(h), Qt.AlignCenter, text]
        painter.drawText(text[0], text[1], text[2], text[3], text[4], text[5])
        # painter.restore()

    def draw_big_needle_center_point(self, diameter=30):
        painter = QPainter(self)
        # painter.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.Antialiasing)

        # Koordinatenursprung in die Mitte der Flaeche legen
        painter.translate(self.width() / 2, self.height() / 2)
        painter.setPen(Qt.NoPen)
        # painter.setPen(Qt.NoPen)
        painter.setBrush(self.CenterPointColor)
        # diameter = diameter # self.widget_diameter/6
        painter.drawEllipse(int(-diameter / 2), int(-diameter / 2), int(diameter), int(diameter))

    def draw_needle(self):
        painter = QPainter(self)
        # painter.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        painter.setRenderHint(QPainter.Antialiasing)
        # Koordinatenursprung in die Mitte der Flaeche legen
        painter.translate(self.width() / 2, self.height() / 2)
        painter.setPen(Qt.NoPen)
        painter.setBrush(self.NeedleColor)
        painter.rotate(((self.value - self.value_offset - self.value_min) * self.scale_angle_size /
                        (self.value_max - self.value_min)) + 90 + self.scale_angle_start_value)

        painter.drawConvexPolygon(self.value_needle[0])

    ###############################################################################################
    # Events
    ###############################################################################################

    def resizeEvent(self, event):
        # self.resized.emit()
        # return super(self.parent, self).resizeEvent(event)
        # print("resized")
        # print(self.width())
        self.rescale_method()
        # self.emit(QtCore.SIGNAL("resize()"))
        # print("resizeEvent")

    def paintEvent(self, event):
        # Main Drawing Event:
        # Will be executed on every change
        # vgl http://doc.qt.io/qt-4.8/qt-demos-affine-xform-cpp.html
        # print("event", event)

        # colored pie area
        if self.enable_filled_Polygon:
            self.draw_filled_polygon()

        # draw scale marker lines
        if self.enable_fine_scaled_marker:
            self.create_fine_scaled_marker()
        if self.enable_big_scaled_marker:
            self.draw_big_scaled_markter()

        # draw scale marker value text
        if self.enable_scale_text:
            self.create_scale_marker_values_text()

        # Display Value
        if self.enable_value_text:
            self.create_values_text()

        # draw needle 1
        if self.enable_Needle_Polygon:
            self.draw_needle()

        # Draw Center Point
        if self.enable_CenterPoint:
            self.draw_big_needle_center_point(diameter=(self.widget_diameter / 6))

    ###############################################################################################
    # MouseEvents
    ###############################################################################################

    def setMouseTracking(self, flag):
        def recursive_set(parent):
            for child in parent.findChildren(QObject):
                try:
                    child.setMouseTracking(flag)
                except:
                    pass
                recursive_set(child)

        QWidget.setMouseTracking(self, flag)
        recursive_set(self)

    def mouseReleaseEvent(self, QMouseEvent):
        # print("released")
        self.NeedleColor = self.NeedleColorReleased

        if not self.use_timer_event:
            self.update()
        pass

    def mouseMoveEvent(self, event):
        x, y = event.x() - (self.width() / 2), event.y() - (self.height() / 2)
        if not x == 0:
            angle = math.atan2(y, x) / math.pi * 180
            # winkellaenge der anzeige immer positiv 0 - 360deg
            # min wert + umskalierter wert
            value = (float(math.fmod(angle - self.scale_angle_start_value + 720, 360)) / \
                     (float(self.scale_angle_size) / float(self.value_max - self.value_min))) + self.value_min
            temp = value
            fmod = float(math.fmod(angle - self.scale_angle_start_value + 720, 360))
            state = 0
            if (self.value - (self.value_max - self.value_min) * self.value_needle_snapzone) <= \
                    value <= \
                    (self.value + (self.value_max - self.value_min) * self.value_needle_snapzone):
                self.NeedleColor = self.NeedleColorDrag
                # todo: evtl ueberpruefen
                #
                state = 9
                # if value >= self.value_max and self.last_value < (self.value_max - self.value_min) / 2:
                if value >= self.value_max and self.last_value < (self.value_max - self.value_min) / 2:
                    state = 1
                    value = self.value_max
                    self.last_value = self.value_min
                    self.valueChanged.emit(int(value))
                elif value >= self.value_max >= self.last_value:
                    state = 2
                    value = self.value_max
                    self.last_value = self.value_max
                    self.valueChanged.emit(int(value))
                else:
                    state = 3
                    self.last_value = value
                    self.valueChanged.emit(int(value))

                # todo: mouse event debug output

                # self.update_value(value, mouse_controlled=True)

                # self.valueChanged.emit(int(value))
                # print(str(int(value)))
            # self.valueChanged.emit()

            # todo: convert print to logging debug
            # print('mouseMoveEvent: x=%d, y=%d, a=%s, v=%s, fmod=%s, temp=%s, state=%s' % (
            #     x, y, angle, value, fmod, temp, state))




    # def createPoly(self, n, r, s):
    #     polygon = QPolygonF()
    #
    #     w = 360/n                                                       # angle per step
    #     for i in range(n):                                              # add the points of polygon
    #         t = w*i + s
    #         x = r*math.cos(math.radians(t))
    #         y = r*math.sin(math.radians(t))
    #         # polygon.append(QtCore.QPointF(self.width()/2 +x, self.height()/2 + y))
    #         polygon.append(QtCore.QPointF(x, y))
    #
    #     return polygon




################################################################################################
# DEMO Routine
# required: analoggaugewidget_demo.ui
# compile analoggaugewidget_demo.ui -> analoggaugewidget_demo_ui.py
# show a lot of variables and possibilities formodification
################################################################################################
if __name__ == '__main__':
    def main():
        import sys
        app = QApplication(sys.argv)
        my_gauge = AnalogGaugeWidget()
        my_gauge.show()
        sys.exit(app.exec_())

    class mainclass():

        def __init__(self):
            import os  # Used in Testing Script
            import sys

            if used_Qt_Version == 4:
                print("Compile QUI for Qt Version: " + str(used_Qt_Version))
                os.system("pyuic4 -o analoggaugewidget_demo_ui.py analoggaugewidget_demo.ui")
            elif used_Qt_Version == 5:
                print("Compile QUI for Qt Version: " + str(used_Qt_Version))
                os.system("pyuic5 -o analoggaugewidget_demo_ui.py analoggaugewidget_demo.ui")

            from analoggaugewidget_demo_ui import Ui_MainWindow

            self.app = QApplication(sys.argv)
            window = QMainWindow()
            self.my_gauge = Ui_MainWindow()
            self.my_gauge.setupUi(window)
            window.show()
            self.my_gauge.widget.enable_barGraph = True

            self.my_gauge.widget.value_needle_snapzone = 1

            self.my_gauge.widget.value_min = 0
            self.my_gauge.widget.value_max = 1100
            self.my_gauge.widget.scala_main_count = 11
            self.my_gauge.ActualSlider.setMaximum(self.my_gauge.widget.value_max)
            self.my_gauge.ActualSlider.setMinimum(self.my_gauge.widget.value_min)
            self.my_gauge.AussenRadiusSlider.setValue(self.my_gauge.widget.gauge_color_outer_radius_factor * 1000)
            self.my_gauge.InnenRadiusSlider.setValue(self.my_gauge.widget.gauge_color_inner_radius_factor * 1000)

            self.my_gauge.GaugeStartSlider.setValue(self.my_gauge.widget.scale_angle_start_value)
            self.my_gauge.GaugeSizeSlider.setValue(self.my_gauge.widget.scale_angle_size)

            # set Start Value
            # self.my_gauge.widget.update_value(self.my_gauge.widget.value_min)
            self.my_gauge.widget.update_value(int(self.my_gauge.widget.value_max - self.my_gauge.widget.value_min)/2)

            ################################################################################################
            # Anzeigenadel Farbe anpassen
            # auf Slider Aenderung reagieren
            self.my_gauge.RedSlider_Needle.valueChanged.connect(self.set_NeedleColor)
            self.my_gauge.GreenSlider_Needle.valueChanged.connect(self.set_NeedleColor)
            self.my_gauge.BlueSlider_Needle.valueChanged.connect(self.set_NeedleColor)
            self.my_gauge.TrancSlider_Needle.valueChanged.connect(self.set_NeedleColor)

            # LCD Default Werte festlegen
            self.my_gauge.lcdNumber_Red_Needle.display(self.my_gauge.RedSlider_Needle.value())
            self.my_gauge.lcdNumber_Green_Needle.display(self.my_gauge.GreenSlider_Needle.value())
            self.my_gauge.lcdNumber_Blue_Needle.display(self.my_gauge.RedSlider_Needle.value())
            self.my_gauge.lcdNumber_Trancparency_Needle.display(self.my_gauge.TrancSlider_Needle.value())

            ################################################################################################
            # Anzeigenadel Farbe anpassen bei manueller Beswegung
            # auf Slider Aenderung reagieren
            self.my_gauge.RedSlider_NeedleDrag.valueChanged.connect(self.set_NeedleColorDrag)
            self.my_gauge.GreenSlider_NeedleDrag.valueChanged.connect(self.set_NeedleColorDrag)
            self.my_gauge.BlueSlider_NeedleDrag.valueChanged.connect(self.set_NeedleColorDrag)
            self.my_gauge.TrancSlider_NeedleDrag.valueChanged.connect(self.set_NeedleColorDrag)

            # LCD Default Werte festlegen
            self.my_gauge.lcdNumber_Red_NeedleDrag.display(self.my_gauge.RedSlider_NeedleDrag.value())
            self.my_gauge.lcdNumber_Green_NeedleDrag.display(self.my_gauge.GreenSlider_NeedleDrag.value())
            self.my_gauge.lcdNumber_Blue_NeedleDrag.display(self.my_gauge.BlueSlider_NeedleDrag.value())
            self.my_gauge.lcdNumber_Trancparency_NeedleDrag.display(self.my_gauge.TrancSlider_NeedleDrag.value())


            ################################################################################################
            # Skala Text Farbe anpassen
            # auf Slider Aenderung reagieren
            self.my_gauge.RedSlider_Scale.valueChanged.connect(self.set_ScaleValueColor)
            self.my_gauge.GreenSlider_Scale.valueChanged.connect(self.set_ScaleValueColor)
            self.my_gauge.BlueSlider_Scale.valueChanged.connect(self.set_ScaleValueColor)
            self.my_gauge.TrancSlider_Scale.valueChanged.connect(self.set_ScaleValueColor)

            # LCD Default Werte festlegen
            self.my_gauge.lcdNumber_Red_Scale.display(self.my_gauge.RedSlider_Scale.value())
            self.my_gauge.lcdNumber_Green_Scale.display(self.my_gauge.GreenSlider_Scale.value())
            self.my_gauge.lcdNumber_Blue_Scale.display(self.my_gauge.BlueSlider_Scale.value())
            self.my_gauge.lcdNumber_Trancparency_Scale.display(self.my_gauge.TrancSlider_Scale.value())


            ################################################################################################
            # Display Text Farbe anpassen
            # auf Slider Aenderung reagieren
            self.my_gauge.RedSlider_Display.valueChanged.connect(self.set_DisplayValueColor)
            self.my_gauge.GreenSlider_Display.valueChanged.connect(self.set_DisplayValueColor)
            self.my_gauge.BlueSlider_Display.valueChanged.connect(self.set_DisplayValueColor)
            self.my_gauge.TrancSlider_Display.valueChanged.connect(self.set_DisplayValueColor)

            # LCD Default Werte festlegen
            self.my_gauge.lcdNumber_Red_Display.display(self.my_gauge.RedSlider_Display.value())
            self.my_gauge.lcdNumber_Green_Display.display(self.my_gauge.GreenSlider_Display.value())
            self.my_gauge.lcdNumber_Blue_Display.display(self.my_gauge.BlueSlider_Display.value())
            self.my_gauge.lcdNumber_Trancparency_Display.display(self.my_gauge.TrancSlider_Display.value())


            self.my_gauge.CB_barGraph.stateChanged.connect(self.en_disable_barGraph)
            self.my_gauge.CB_ValueText.stateChanged.connect(self.en_disable_ValueText)
            self.my_gauge.CB_CenterPoint.stateChanged.connect(self.en_disable_CB_CenterPoint)
            self.my_gauge.CB_ScaleText.stateChanged.connect(self.en_disable_ScaleText)
            self.my_gauge.CB_ShowBarGraph.stateChanged.connect(self.set_enable_filled_Polygon)


            self.my_gauge.CB_Grid.stateChanged.connect(self.set_enable_Scale_Grid)
            self.my_gauge.CB_fineGrid.stateChanged.connect(self.set_enable_fine_Scale_Grid)

            self.my_gauge.CB_Needle.stateChanged.connect(self.en_disable_Needle)

            # my_gauge.widget.set_scale_polygon_colors([[.0, Qt.transparent]])
            # my_gauge.widget.set_scale_polygon_colors([[.0, Qt.yellow]])
            # my_gauge.widget.set_scale_polygon_colors(None)
            # my_gauge.widget.enable_filled_Polygon = False
            sys.exit(self.app.exec_())

        def set_NeedleColor(self):
            #print(self.my_gauge.RedSlider.value())
            R = self.my_gauge.RedSlider_Needle.value()
            G = self.my_gauge.GreenSlider_Needle.value()
            B = self.my_gauge.BlueSlider_Needle.value()
            Transparency = self.my_gauge.TrancSlider_Needle.value()
            # print(R, G, B, Transparency)
            self.my_gauge.widget.set_NeedleColor(R=R, G=G, B=B, Transparency=Transparency)

        def set_NeedleColorDrag(self):
            #print(self.my_gauge.RedSlider.value())
            R = self.my_gauge.RedSlider_NeedleDrag.value()
            G = self.my_gauge.GreenSlider_NeedleDrag.value()
            B = self.my_gauge.BlueSlider_NeedleDrag.value()
            Transparency = self.my_gauge.TrancSlider_NeedleDrag.value()
            # print(R, G, B, Transparency)
            self.my_gauge.widget.set_NeedleColorDrag(R=R, G=G, B=B, Transparency=Transparency)


        def set_ScaleValueColor(self):
            #print(self.my_gauge.RedSlider.value())
            R = self.my_gauge.RedSlider_Scale.value()
            G = self.my_gauge.GreenSlider_Scale.value()
            B = self.my_gauge.BlueSlider_Scale.value()
            Transparency = self.my_gauge.TrancSlider_Scale.value()
            # print(R, G, B, Transparency)
            self.my_gauge.widget.set_ScaleValueColor(R=R, G=G, B=B, Transparency=Transparency)

        def set_DisplayValueColor(self):
            # print(self.my_gauge.RedSlider.value())
            R = self.my_gauge.RedSlider_Display.value()
            G = self.my_gauge.GreenSlider_Display.value()
            B = self.my_gauge.BlueSlider_Display.value()
            Transparency = self.my_gauge.TrancSlider_Display.value()
            # print(R, G, B, Transparency)
            self.my_gauge.widget.set_DisplayValueColor(R=R, G=G, B=B, Transparency=Transparency)

        def en_disable_barGraph(self):
            self.my_gauge.widget.set_enable_barGraph(self.my_gauge.CB_barGraph.isChecked())

        def en_disable_ValueText(self):
            self.my_gauge.widget.set_enable_value_text(self.my_gauge.CB_ValueText.isChecked())

        def en_disable_CB_CenterPoint(self):
            self.my_gauge.widget.set_enable_CenterPoint(self.my_gauge.CB_CenterPoint.isChecked())

        def en_disable_Needle(self):
            self.my_gauge.widget.set_enable_Needle_Polygon(self.my_gauge.CB_Needle.isChecked())

        def en_disable_ScaleText(self):
            self.my_gauge.widget.set_enable_ScaleText(self.my_gauge.CB_ScaleText.isChecked())

        def set_enable_filled_Polygon(self):
            self.my_gauge.widget.set_enable_filled_Polygon(self.my_gauge.CB_ShowBarGraph.isChecked())

        def set_enable_Scale_Grid(self):
            self.my_gauge.widget.set_enable_big_scaled_grid(self.my_gauge.CB_Grid.isChecked())

        def set_enable_fine_Scale_Grid(self):
            self.my_gauge.widget.set_enable_fine_scaled_marker(self.my_gauge.CB_fineGrid.isChecked())


    ############################################
    # Run DEMO Routine
    ############################################
    main = mainclass()