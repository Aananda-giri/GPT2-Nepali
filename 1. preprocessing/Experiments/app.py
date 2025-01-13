from flask import Flask, render_template_string, request, redirect, url_for
import pickle
import math

app = Flask(__name__)

# Sample data format: {key: [text, index_to_highlight]}
data = {
    f'key{i}': [f"Sample text data {i}. This is a long text entry for demonstration purposes.", i % 30]
    for i in range(1, 201)  # Example with 200 entries
}

with open('cleaned_char_in_text.pkl', 'rb') as f:
    data = pickle.load(f)

# Dictionary to store characters to replace
chars_to_replace = {}

# Function to save the chars_to_replace dictionary to a pickle file
def save_replacements():
    with open('chars_to_replace.pkl', 'wb') as f:
        pickle.dump(chars_to_replace, f)

def highlight_text(text, index):
    """Highlights the character at the specified index in the given text."""
    if index < 0 or index >= len(text):
        return text  # Return the original text if the index is out of bounds

    # Wrap the character at the specified index with a <span> tag for highlighting
    highlighted_text = (
        text[:index] +
        '<span style="background-color: yellow; font-weight: bold;">' + text[index] + '</span>' +
        text[index + 1:]
    )
    return highlighted_text

@app.route('/', methods=['GET', 'POST'])
def display_data():
    """Endpoint to display paginated text entries with highlighted characters and a form for replacement."""
    global chars_to_replace

    if request.method == 'POST':
        # Get the key, new character, and current page from the form submission
        key = request.form['key']
        new_char = request.form['new_char']
        page = request.form.get('page', 1)  # Default to page 1 if not provided
        original_char = data[key][0][data[key][1]]  # Original character based on the index

        if original_char != new_char:
            # Update the dictionary with the replacement
            chars_to_replace[original_char] = new_char

            # Save the updated dictionary to a pickle file
            save_replacements()

            # # Redirect to the same page to show the updated form
            # return redirect(url_for('display_data', page=page))

    # Pagination logic
    page = int(request.args.get('page', 1))  # Get the current page number from query params
    per_page = 50  # Number of items per page
    total_entries = len(data)
    total_pages = math.ceil(total_entries / per_page)

    # Get the start and end indices for the current page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = list(data.items())[start:end]

    # Prepare highlighted entries for display
    highlighted_entries = {key: (highlight_text(text, index), index) for key, (text, index) in paginated_data}

    # Render the result using an HTML template
    html_template = '''
    <!DOCTYPE html>
<html>
<head>
    <title>Highlighted Text Data</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            font-size: 18px;
            line-height: 1.6;
        }
        h2 {
            color: #333;
        }
        .entry {
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
            background-color: #f9f9f9;
        }
        .replace-form {
            margin-top: 10px;
        }
        .pagination {
            margin-top: 20px;
            text-align: center;
        }
        .pagination a {
            margin: 0 5px;
            text-decoration: none;
            padding: 8px 16px;
            border: 1px solid #ccc;
            border-radius: 5px;
            color: #333;
        }
        .pagination a.active {
            background-color: #4CAF50;
            color: white;
            border: 1px solid #4CAF50;
        }
    </style>
    <script>
        // Save scroll position before form submission
        function saveScrollPosition() {
            localStorage.setItem('scrollPosition', window.scrollY);
        }

        // Restore scroll position after page load
        window.onload = function() {
            const scrollPosition = localStorage.getItem('scrollPosition');
            if (scrollPosition) {
                window.scrollTo(0, parseInt(scrollPosition));
                localStorage.removeItem('scrollPosition');
            }
        }
    </script>
</head>
<body>
    <h1>Highlighted Text Entries (Page {{ page }} of {{ total_pages }})</h1>
    {% for key, (highlighted_text, index) in highlighted_entries.items() %}
        <div class="entry">
            <h2>Key: "{{ key }}" (Index: {{ index }})</h2>
            <p>{{ highlighted_text|safe }}</p>
            <form method="post" class="replace-form" onsubmit="saveScrollPosition()">
                <input type="hidden" name="key" value="{{ key }}">
                <input type="hidden" name="page" value="{{ page }}">
                <label for="new_char">Replace highlighted character with:</label>
                <input type="text" name="new_char" maxlength="2" required>
                <button type="submit">Replace</button>
            </form>
        </div>
    {% endfor %}

    <div class="pagination">
        {% if page > 1 %}
            <a href="{{ url_for('display_data', page=page-1) }}">Previous</a>
        {% endif %}
        {% for p in range(1, total_pages + 1) %}
            <a href="{{ url_for('display_data', page=p) }}" class="{{ 'active' if p == page else '' }}">{{ p }}</a>
        {% endfor %}
        {% if page < total_pages %}
            <a href="{{ url_for('display_data', page=page+1) }}">Next</a>
        {% endif %}
    </div>
</body>
</html>

    '''
    return render_template_string(html_template, highlighted_entries=highlighted_entries, page=page, total_pages=total_pages)

if __name__ == '__main__':
    app.run(debug=True)
