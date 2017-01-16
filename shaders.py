from collections import namedtuple

_Shader = namedtuple('Shader', 'vertex fragment')

IMAGE = _Shader("""
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
""", """
    uniform sampler2D texture;
    varying vec2 v_texcoord;
    void main()
    {
        gl_FragColor = texture2D(texture, v_texcoord);
    }
""")

COIN = _Shader("""
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
""", """
    uniform vec4 circle_color;
    uniform vec4 border_color;
    uniform vec4 bkg_color;
    varying vec2 v_texcoord;
    void main()
    {
        float dist = sqrt(dot(v_texcoord, v_texcoord));
        if (dist < 0.9)
            // inside the border
            if (abs(v_texcoord.x) < 0.1 && abs(v_texcoord.y) < 0.5)
                // draw a vertical slot
                gl_FragColor = border_color;
            else
                gl_FragColor = circle_color;
        else if (dist < 1)
            // the border
            gl_FragColor = border_color;
        else
            // outside the border
            gl_FragColor = bkg_color;
    }
""")

AGENT = _Shader("""
    attribute vec2 position;
    attribute vec2 texcoord;
    varying vec2 v_texcoord;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_texcoord = texcoord;
    }
""", """
    uniform vec4 circle_color;
    uniform vec4 pointer_color;
    uniform vec4 border_color;
    uniform vec4 bkg_color;
    uniform float pointer_threshold;
    uniform float theta;
    varying vec2 v_texcoord;
    void main()
    {
        float dist = sqrt(dot(v_texcoord, v_texcoord));
        if (dist < 0.9)
        {
            // inside the border; calculate angle to draw pointer
            vec2 coord_unit = v_texcoord / dist;
            float theta_actual = atan(coord_unit.y, coord_unit.x);
            float theta_diff = theta - theta_actual;
            float bounded_diff = abs(atan(sin(theta_diff), cos(theta_diff)));
            if (bounded_diff > pointer_threshold)
                // outside the pointer arc
                gl_FragColor = circle_color;
            else
                // inside the pointer arc
                gl_FragColor = pointer_color;
        }
        else if (dist < 1)
        {
            // the border
            gl_FragColor = border_color;
        }
        else
        {
            // outside the border
            gl_FragColor = bkg_color;
        }
    }
""")

LINE = _Shader("""
    attribute vec2 position;
    attribute vec4 line_color;
    varying vec4 v_line_color;
    void main()
    {
        gl_Position = vec4(position, 0.0, 1.0);
        v_line_color = line_color;
    }
""", """
    varying vec4 v_line_color;
    void main()
    {
        gl_FragColor = v_line_color;
    }
""")
