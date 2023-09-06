# Import necessary libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import spacy
import nltk
from nltk.tokenize import word_tokenize
import dash_bootstrap_components as dbc  # Import dbc for Bootstrap styling

# Initialize spaCy model for text similarity
nlp = spacy.load("en_core_web_sm")

# Create a Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)  # Apply Bootstrap theme

# Define colors for the website
colors = {
    'background': '#E8FFCE',   # Background color
    'text': '#071952',         # Text color
    'h_text': '#2D2727',       # Heading text color
    'b_background': '#053B50', # Button background color
    'b_t_color': '#F5F5F5',    # Button text color
    'q_color': '#27005D',      # Question text color
    'a_color': '##3F2305'      # Answer text color
}

# Define the navigation bar
navbar = dbc.NavbarSimple(
    children=[
        dbc.NavItem(dbc.NavLink("Home", href="/")),
        dbc.NavItem(dbc.NavLink("Cossine Calculator", href="/c_calculator")),
        dbc.NavItem(dbc.NavLink("Jaccard Calculator", href="/j_calculator")),
        dbc.NavItem(dbc.NavLink("About Us", href="/about")),
    ],
    brand="Calculator",
    brand_href="/",
    color="#040D12",  # Navbar color
    dark=True,
)

# Define the app layout with styles
app.layout = html.Div(style={'backgroundColor': colors['background'], 'padding': '20px'}, children=[
    dcc.Location(id='url', refresh=False),  # Location for handling URL changes
    
    navbar,  # Add the navigation bar
    
    html.Div(id='page-content')
])

# Define callback to update the page content based on the URL
@app.callback(Output('page-content', 'children'), Input('url', 'pathname'))
def display_page(pathname):
    if pathname == '/c_calculator':
        return cossine_calculator_layout()  # Display calculator page
    if pathname == '/j_calculator':
        return jaccard_calculator_layout()  # Display calculator page
    elif pathname == '/about':
        return about_layout()  # Display about page
    else:
        return home_layout()  # Display home page (default)

# Define home page layout
def home_layout():
    return html.Div([
        html.H1("Welcome to the Text Similarity Calculator", style={'textAlign': 'center', 'color': colors['text']}),
        html.H3("What is Cosine Similarity?", style={'color': colors['q_color']}),
        # Explanation of cosine similarity
        html.P("=> Cosine similarity measures the similarity between two vectors of an inner product space. It is measured by the cosine of the angle between two vectors and determines whether two vectors are pointing in roughly the same direction. It is often used to measure document similarity in text analysis.", style={'color': colors['a_color']}),
        
        html.H3("Cosine Similarity Formula:", style={'color': colors['q_color']}),
        # Explanation of the cosine similarity formula
        html.P("Cosine similarity calculates the cosine of the angle between two vectors. The formula for cosine similarity between vectors A and B is:", style={'color': colors['a_color']}),
        html.P("cosine_similarity(A, B) = (A dot B) / (||A|| * ||B||)", style={'color': colors['a_color']}),

        html.H3("What is Jaccard Similarity?", style={'color': colors['q_color']}),
        # Explanation of Jaccard similarity
        html.P("Jaccard Similarity is a common proximity measurement used to compute the similarity between two objects, such as two text documents. Jaccard similarity can be used to find the similarity between two asymmetric binary vectors or to find the similarity between two sets. In literature, Jaccard similarity, symbolized by,"
"can also be referred to as Jaccard Index, Jaccard Coefficient, Jaccard Dissimilarity, and Jaccard Distance.", style={'color': colors['a_color']}),

        html.H3("Jaccard similarity formula:", style={'color': colors['q_color']}),
       # Explanation of the Jaccard similarity formula
        html.P("Jaccard Similarity (J) = (Size of Intersection) / (Size of Union)", style={'color': colors['a_color']}),
    ])

# Define calculator page layout for Cosine Similarity
def cossine_calculator_layout():
    return html.Div([
        html.H1("Cosine Similarity Calculator", style={'textAlign': 'center', 'color': colors['text']}),
        
        html.H3("Enter Your First Text:", style={'color': colors['h_text']}),
        # Text input for the first text
        dcc.Textarea(id='text1', placeholder="First Text...", style={'width': '100%', 'height': '70px', 'marginBottom': '20px'}),
        html.H3("Enter Your Second Text:", style={'color': colors['h_text']}),
        # Text input for the second text
        dcc.Textarea(id='text2', placeholder="Second Text ...", style={'width': '100%', 'height': '70px', 'marginBottom': '20px'}),
        
        # Button to calculate similarity
        html.Button('Calculate Similarity', id='c-calculate-button', n_clicks=0, style={'margin-top': '20px','fontSize': '26px', 'backgroundColor': colors['b_background'], 'color': colors['b_t_color'], 'border': 'none'}),
        
        # Display similarity score
        html.Div(id='output-cosine', style={'textAlign': 'center','margin-top': '20px','marginBottom': '40px', 'fontSize': '40px', 'fontWeight': 'bold', 'color': '#1A5D1A'}),
    ])

# Define calculator page layout for Jaccard Similarity
def jaccard_calculator_layout():
    return html.Div([
        html.H1("Jaccard Similarity Calculator", style={'textAlign': 'center', 'color': colors['text']}),
        
        html.H3("Enter Your First Text:", style={'color': colors['h_text']}),
        # Text input for the first text
        dcc.Textarea(id='text3', placeholder="First Text...", style={'width': '100%', 'height': '70px', 'marginBottom': '20px'}),
        html.H3("Enter Your Second Text:", style={'color': colors['h_text']}),
        # Text input for the second text
        dcc.Textarea(id='text4', placeholder="Second Text ...", style={'width': '100%', 'height': '70px', 'marginBottom': '20px'}),
        
        # Button to calculate similarity
        html.Button('Calculate Similarity', id='j-calculate-button', n_clicks=0, style={'margin-top': '20px','fontSize': '26px', 'backgroundColor': colors['b_background'], 'color': colors['b_t_color'], 'border': 'none'}),
        
        # Display similarity score
        html.Div(id='output-jaccard', style={'textAlign': 'center','margin-top': '20px','marginBottom': '40px', 'fontSize': '40px', 'fontWeight': 'bold', 'color': '#1A5D1A'}),
    ])


# Define about page layout
def about_layout():
    return html.Div([
        html.H1("About Us", style={'textAlign': 'center', 'color': colors['text']}),
        # Image (add your image source here)
        html.Img(src="about_image.jpg", width="100%"),
        # About us text
        html.P("👋🏽Hello, My name is Md. Samiul Islam! Nickname: Sam", style={'textAlign': 'justify', 'color' : colors['a_color']}),
        html.P("🎓A Computer science and engineering student with a strong interest in Machine learning and Data science. I am an ambitious and passionate individual who is focused on conducting research in these fields. My background in Computer science, along with my knowledge of Machine learning and Data science techniques, has given me a solid foundation to excel in this field.", style={'textAlign': 'justify', 'color' : colors['a_color']}),
    ])

# Define a callback function to calculate and display text similarity for Cosine Similarity
@app.callback(
    Output('output-cosine', 'children'),
    Input('c-calculate-button', 'n_clicks'),
    Input('text1', 'value'),
    Input('text2', 'value')
)
def calculate_cossine_similarity(n_clicks, text1, text2):
    if n_clicks > 0 and text1 and text2:
        # Process the text using spaCy
        doc1 = nlp(text1)
        doc2 = nlp(text2)
        
        # Calculate similarity (cosine similarity)
        similarity_score = doc1.similarity(doc2)
        
        return f"Similarity Score: {similarity_score:.2f}"
    else:
        return ""
    
# Define a callback function to calculate and display text similarity for Jaccard Similarity
@app.callback(
    Output('output-jaccard', 'children'),
    Input('j-calculate-button', 'n_clicks'),
    Input('text3', 'value'),
    Input('text4', 'value')
)
def calculate_jaccard_similarity(n_clicks, text3, text4):
    if n_clicks > 0 and text3 and text4:
        # Tokenize the input texts
        words1 = set(word_tokenize(text3.lower()))
        words2 = set(word_tokenize(text4.lower()))
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1) + len(words2) - intersection
        jaccard_similarity = intersection / union
        
        return f"Jaccard Similarity Score: {jaccard_similarity:.2f}"
    else:
        return ""


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
