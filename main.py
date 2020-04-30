from flask import Flask, request, render_template, session, redirect
import cbrs

app = Flask(__name__)
book_titles = cbrs.titles()

@app.route('/')
def inputBook():
    return render_template('index.html', datas=book_titles)


@app.route('/', methods=['POST'])
def recommender():
    book_title = request.form['book_title']
    df = cbrs.recommend(book_title)
    # return render_template('index.html', tables=[df.to_html(classes='data')], titles=df.columns.values, datas=book_titles, chosen=book_title)
    return render_template('index.html', tables=df.values, titles=df.columns.values, datas=book_titles, chosen=book_title)


if __name__ == '__main__':
    app.run(debug=True)