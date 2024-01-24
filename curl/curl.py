import numpy as np
import sympy as sym
import random 
import re

from manim import *
from sympy.vector import CoordSys3D, Del
from sympy import latex
        
        
class Curl(Scene):
    """
    Manim scene that takes a 2D vector function, gets its curl, plots its vector field, and draws a dot 
    that moves through space displaying the curl at that point
    """
    def construct(self):
        
        def get_latex(func):
            """ returns formatted and cleaned latex to use for func """
            tex = latex(func).replace('}}', '} }').replace('{{', '{ {')
            tex = re.sub(r'_{R}', '', tex)
            tex = re.sub(r'mathbf', 'boldsymbol', tex)
            return tex
        
        def min_max_curl_value():
            x_values = np.linspace(-5, 5, 200)
            y_values = np.linspace(-5, 5, 200)
            
            X, Y = np.meshgrid(x_values, y_values)
            curl_values = curl_lambda(X, Y)
            return np.min(curl_values), np.max(curl_values)
        
        def min_max_func_value():
            x_values = np.linspace(-5, 5, 200)
            y_values = np.linspace(-5, 5, 200)

            X, Y = np.meshgrid(x_values, y_values)
            func_values = np.sqrt(func_x_lambda(X, Y)**2 + func_y_lambda(X, Y)**2)
            return np.min(func_values), np.max(func_values)
        
        
        R = CoordSys3D('R')
        delop = Del()
        
        # input vector function (sympy functions should be used such as sym.sin())
        vec_func = R.x**2 * sym.sin(R.y) * R.i \
                 + R.y**2 * -sym.cos(R.x) * R.j 
    
        # latex of the function
        func_tex = get_latex(vec_func)
        func_text = MathTex("\\vec V(x, y) = " + func_tex).set_color_by_gradient(BLUE, PURPLE, RED).set_height(1).move_to(ORIGIN)
        func_bg = BackgroundRectangle(func_text, color=BLACK, fill_opacity=0.6)

        curl_func = delop.cross(vec_func).doit()
        
        # latex of the curl of the function
        curl_tex = get_latex(curl_func)
        curl_func_text = Tex("$\\nabla \\times \\vec V = " + curl_tex +'$').set_color_by_gradient(BLUE, PURPLE, RED).set_height(1).move_to(ORIGIN)
        curl_bg = BackgroundRectangle(curl_func_text, color=BLACK, fill_opacity=0.6)
        
        # lambda function definitions
        func_x_lambda = sym.lambdify(args=[R.x, R.y], expr=vec_func.dot(R.i), modules='numpy')
        func_y_lambda = sym.lambdify(args=[R.x, R.y], expr=vec_func.dot(R.j), modules='numpy')
        
        curl_lambda = sym.lambdify(args=[R.x, R.y], expr=curl_func.dot(R.k), modules='numpy')
        
        # animation lambda function
        func = lambda pos: func_x_lambda(pos[0], pos[1]) * RIGHT + func_y_lambda(pos[0], pos[1]) * UP
        
        dot = Dot(fill_color=GREEN_A, fill_opacity=0.8).scale(0.5)
        
        # curl updater to keep updating curl value with respect to dot position
        curl_text = always_redraw(lambda : 
            MathTex(
                f"\\nabla \\times \\vec V = {curl_lambda(dot.get_center()[0], dot.get_center()[1]):.3f}"
                    ).set_color_by_gradient(BLUE, PURPLE, RED).set_height(0.25)
        )
        curl_text.next_to(dot, direction=UP+RIGHT, buff=SMALL_BUFF)
        curl_text.add_updater(lambda text: text.next_to(dot, direction=UP+RIGHT, buff=SMALL_BUFF))
        
        nplane=NumberPlane()
        nplane.add_coordinates()
        
        func_boundary = min_max_func_value()
        curl_boundary = min_max_curl_value()
        
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
        vector_field = ArrowVectorField(
            func,
            min_color_scheme_value=func_boundary[0],
            max_color_scheme_value=func_boundary[1], 
            colors=colors,
            )
        
        stream_lines = StreamLines(
            func,
            min_color_scheme_value=func_boundary[0],
            max_color_scheme_value=func_boundary[1],
            colors=colors,
            stroke_width=2,
            virtual_time=3,
            max_anchors_per_line=90,
            )
        
        
        self.play(Write(func_text))
        self.wait(1)
        self.play(func_text.animate.scale(0.25).to_edge(UL))
        self.play(FadeIn(func_bg))
        
        self.play(Write(curl_func_text))
        self.wait(1)
        self.play(curl_func_text.animate.scale(0.25).to_edge(UR))
        self.play(FadeIn(curl_bg))
        
        self.play(Write(nplane))
        self.play(Create(vector_field))
        self.play(Create(dot))
        self.play(Create(curl_text))
        
        self.play(MoveAlongPath(
            dot,
            ArcBetweenPoints(dot.get_center(), dot.get_center() + PI/2 * RIGHT, angle=0),
            rate_func=smooth
        ))
        
        stream_lines.start_animation(warm_up=True)
        self.add(stream_lines)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
        self.play(stream_lines.end_animation())
        
        dirs = [UP, LEFT, DOWN, RIGHT]
        # Move the dot 5 times
        for i in range(6):
                    
            random.seed(2024)
    
            # Generate a random vector
            vector = random.uniform(3, 4) * (dirs[i % 4] + dirs[(i+1) % 4])
            
            # Create a curved path to the new position
            path = ArcBetweenPoints(dot.get_center(), dot.get_center() + vector, angle=PI/2)
                
            # Move the dot along the path
            self.play(MoveAlongPath(dot, path), run_time=path.get_arc_length(), rate_func=linear)
            
        self.wait(3)
        
        
class CurlWithComments(Scene):
    """
    Manim scene that takes a 2D vector function, gets its curl, plots its vector field, and draws a dot 
    that moves through space displaying the curl at that point
    """
    def construct(self):
        
        def get_latex(func):
            """ returns formatted and cleaned latex to use for func """
            tex = latex(func).replace('}}', '} }').replace('{{', '{ {')
            tex = re.sub(r'_{R}', '', tex)
            tex = re.sub(r'mathbf', 'boldsymbol', tex)
            return tex
        
        
        def min_max_func_value():
            x_values = np.linspace(-5, 5, 200)
            y_values = np.linspace(-5, 5, 200)

            X, Y = np.meshgrid(x_values, y_values)
            func_values = np.sqrt(func_x_lambda(X, Y)**2 + func_y_lambda(X, Y)**2)
            return np.min(func_values), np.max(func_values)
        
        
        R = CoordSys3D('R')
        delop = Del()
        
        
        # input vector function (sympy functions should be used such as sym.sin())
        vec_func = (sym.sin(R.y)) * R.i \
                 + (sym.cos(R.x)) * R.j 
    
        # latex of the function
        func_tex = get_latex(vec_func)
        func_text = MathTex("\\vec V(x, y) = " + func_tex).set_color_by_gradient(BLUE, PURPLE, RED).set_height(1).move_to(ORIGIN)
        func_bg = BackgroundRectangle(func_text, color=BLACK, fill_opacity=0.6)

        curl_func = delop.cross(vec_func).doit()
        
        # latex of the curl of the function
        curl_tex = get_latex(curl_func)
        curl_func_text = Tex("$\\nabla \\times \\vec V = " + curl_tex +'$').set_color_by_gradient(BLUE, PURPLE, RED).set_height(1).move_to(ORIGIN)
        curl_bg = BackgroundRectangle(curl_func_text, color=BLACK, fill_opacity=0.6)
        
        # lambda function definitions
        func_x_lambda = sym.lambdify(args=[R.x, R.y], expr=vec_func.dot(R.i), modules='numpy')
        func_y_lambda = sym.lambdify(args=[R.x, R.y], expr=vec_func.dot(R.j), modules='numpy')
        
        curl_lambda = sym.lambdify(args=[R.x, R.y], expr=curl_func.dot(R.k), modules='numpy')
        
        # animation lambda function
        func = lambda pos: func_x_lambda(pos[0], pos[1]) * RIGHT + func_y_lambda(pos[0], pos[1]) * UP
        
        dot = Dot(fill_color=GREEN_A, fill_opacity=0.8).scale(0.5)
        
        # curl updater to keep updating curl value with respect to dot position
        curl_text = always_redraw(lambda : 
            MathTex(
                f"\\nabla \\times \\vec V = {curl_lambda(dot.get_center()[0], dot.get_center()[1]):.3f}"
                    ).set_color_by_gradient(BLUE, PURPLE, RED).set_height(0.25)
        )
        curl_text.next_to(dot, direction=UP+RIGHT, buff=SMALL_BUFF)
        curl_text.add_updater(lambda text: text.next_to(dot, direction=UP+RIGHT, buff=SMALL_BUFF))
        
        nplane=NumberPlane()
        nplane.add_coordinates()
        
        func_boundary = min_max_func_value()
        
        colors = [RED, ORANGE, YELLOW, GREEN, BLUE]
        vector_field = ArrowVectorField(
            func,
            min_color_scheme_value=func_boundary[0],
            max_color_scheme_value=func_boundary[1], 
            colors=colors,
            )
        
        stream_lines = StreamLines(
            func,
            min_color_scheme_value=func_boundary[0],
            max_color_scheme_value=func_boundary[1],
            colors=colors,
            stroke_width=2,
            virtual_time=3,
            max_anchors_per_line=90,
            )
        
        
        self.play(Write(func_text))
        self.wait(1)
        self.play(func_text.animate.scale(0.25).to_edge(UL))
        self.play(FadeIn(func_bg))
        
        self.play(Write(curl_func_text))
        self.wait(1)
        self.play(curl_func_text.animate.scale(0.25).to_edge(UR))
        self.play(FadeIn(curl_bg))
        
        self.play(Write(nplane))
        self.play(Create(vector_field))
        self.play(Create(dot))
        self.play(Create(curl_text))
        
        self.play(MoveAlongPath(
            dot,
            ArcBetweenPoints(dot.get_center(), dot.get_center() + PI/2 * RIGHT, angle=0),
            rate_func=smooth
        ))
        
        negative_curl_comment = Text("Notice that the curl is negative here as the rotation is clockwise \nand the curl vector would point inside the screen"
                                     ).scale(0.33).move_to(curl_text.get_center() + 0.8 * UP +  RIGHT)
        backgroundRectangle1 = BackgroundRectangle(negative_curl_comment, color=BLACK, fill_opacity=0.5)
        
        max_curl_comment = Text("Notice also that the value here is the highest it'll be \nas this is the center of a vortex like region"
                                ).scale(0.33).move_to(curl_text.get_center() + 0.8 * UP +  0.8 * RIGHT)
        backgroundRectangle2 = BackgroundRectangle(max_curl_comment, color=BLACK, fill_opacity=0.5)
        
        self.play(FadeIn(backgroundRectangle1), Write(negative_curl_comment))
        self.wait(3)
        self.play(FadeOut(negative_curl_comment), FadeOut(backgroundRectangle1))
        
        self.wait(1)
        
        self.play(FadeIn(backgroundRectangle2), Write(max_curl_comment))
        
        stream_lines.start_animation(warm_up=True)
        self.add(stream_lines)
        self.wait(stream_lines.virtual_time / stream_lines.flow_speed)
        self.play(stream_lines.end_animation())
        
        self.play(FadeOut(max_curl_comment), FadeOut(backgroundRectangle2))
        
        dirs = [UP, LEFT, DOWN, RIGHT]
        # Move the dot 5 times
        for i in range(6):
                    
            random.seed(2024)
    
            # Generate a random vector
            vector = random.uniform(3, 4) * (dirs[i % 4] + dirs[(i+1) % 4])
            
            # Create a curved path to the new position
            path = ArcBetweenPoints(dot.get_center(), dot.get_center() + vector, angle=PI/2)
            
            if i == 5:
                positive_curl_comment = Text("Notice that the curl is positive here as the rotation\n is counter-clockwise and the curl vector would point\n outside the screen if you apply the right-handed screw"
                                     ).scale(0.33).move_to(curl_text.get_center() + DOWN +  LEFT)
                backgroundRectangle3 = BackgroundRectangle(positive_curl_comment, color=BLACK, fill_opacity=0.5)
                
                self.play(FadeIn(backgroundRectangle3), Write(positive_curl_comment))
                self.wait(3)
                self.play(FadeOut(positive_curl_comment), FadeOut(backgroundRectangle3))
                
            # Move the dot along the path
            self.play(MoveAlongPath(dot, path), run_time=path.get_arc_length(), rate_func=linear)
            
        self.wait(3)
        
        
        
        
class Transition(Scene):
    def construct(self):
        
        text = Text("Now we'll try the same code\n\ton a different function.").set_height(1).move_to(ORIGIN)
        
        self.play(Write(text))
        self.wait(2)
        self.play(Unwrite(text))
        
        
class Opening(Scene):
    def construct(self):
        
        text = """
        In this video, I'll show a simple animation made using manim (a python library)\n
        that draws the vector field of a given 2D function, then uses sympy to calculate\n
        the curl of the given function and draws a point that moves through the vector field\n
        displaying the curl at that position.
        """
        text = Text(text).set_height(1.75).move_to(ORIGIN)
        
        self.play(Write(text))
        self.wait(4)
        self.play(Unwrite(text, reverse=False))
        
        