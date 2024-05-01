# Flask Imports
from flask import Flask, render_template, request, jsonify

# Imports
import googlemaps
import pandas as pd
import numpy as np
import nltk
import os
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

app = Flask(__name__)

@app.route('/')
def index():
    # Added here for faster response later
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('punkt')
    return render_template('index.html')

@app.route('/process_input', methods=['POST'])
def process_text():
    try:
        # Retrieve data from the form
        keyword = request.form.get('keyword')
        location = request.form.get('location')

        # Google API key.
        api_key = os.getenv('API_KEY')

        # Initialize the Google Maps client.
        gmaps = googlemaps.Client(key=api_key)

        def get_reviews(keyword, location):
            # Perform a Places API search based on keyword and location.
            places = gmaps.places(query=keyword, location=location)

            # Extract place details, including reviews.
            places_list = []
            for place in places['results']:
                place_id = place['place_id']
                place_details = gmaps.place(place_id=place_id, fields=['name', 'opening_hours', 'reviews'])
                places_list.append(place_details)

            return places_list

        def extract_reviews(places_list):
            reviews_dict = {}
            for place_details in places_list:
                place_name = place_details['result']['name']
                reviews_list = reviews_dict.get(place_name, [])
                for review in place_details['result'].get('reviews', []):
                    review_text = review.get('text', '')
                    reviews_list.append({'Place Name': place_name, 'Review': review_text})
                reviews_dict[place_name] = reviews_list

            return reviews_dict

        places_list = get_reviews(keyword, location)
        reviews_dict = extract_reviews(places_list)

        output_string = []

        # Add the scale information to the output
        output_string.append("<div style='text-align: left'><h3 style='text-align: center; padding-bottom:1.5rem'>Here's a simplified scale to help you interpret these scores:</h3>")
        output_string.append("<p style='font-weight:bold'>Positive Importance Score:</p>")
        output_string.append("<ul>")
        output_string.append("<li>A high positive score indicates that the word is strongly associated with positive sentiment.</li>")
        output_string.append("<li>The higher the positive score, the more influential the word is in suggesting a positive sentiment in the reviews.</li>")
        output_string.append("</ul>")
        output_string.append("<p style='font-weight:bold'>Negative Importance Score:</p>")
        output_string.append("<ul>")
        output_string.append("<li>A high negative score indicates that the word is strongly associated with negative sentiment.</li>")
        output_string.append("<li>The higher the negative score, the more influential the word is in suggesting a negative sentiment in the reviews.</li>")
        output_string.append("</ul>")
        output_string.append("<p style='font-weight:bold'>Neutral Importance Score:</p>")
        output_string.append("<ul>")
        output_string.append("<li>A score close to zero suggests that the word has a balanced or neutral impact on sentiment.</li>")
        output_string.append("<li>Words with scores around zero may not strongly contribute to either positive or negative sentiment.</li>")
        output_string.append("</ul></div>")

        for place_name, reviews_list in reviews_dict.items():
            df = pd.DataFrame(reviews_list)

            # Data preprocessing
            stop_words = set(stopwords.words('english'))
            custom_stop_words = ["really", "went", "sure", "definitely", "got", "back", "everything", "always",
                                 "first", "second", "third", "omg", "Absolutely", "nice", "much", "make", "store",
                                 "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "1",
                                 "2", "3", "4", "5", "6", "7", "8", "9", "10", "ok", "okay", "friend", "well", "go",
                                 "awesome", "Accordingly", "Actually", "Admirably", "Allegedly", "Almost", "Already",
                                 "Always", "Amazingly", "Approximately", "Arguably", "As", "As a result", "As usual",
                                 "Basically", "Beautifully", "Before", "Clearly", "Completely", "Consequently",
                                 "Constantly", "Definitely", "Deliberately", "Doubtfully", "Eagerly", "Easily",
                                 "Effectively", "Emphatically", "Entirely", "Equally", "Especially", "Essentially",
                                 "Eventually", "Exactly", "Extremely", "Famously", "Finally", "Financially",
                                 "Fortunately", "Frankly", "Freely", "Frequently", "Generally", "Generally speaking",
                                 "Gently", "Gradually", "Happily", "Hardly", "Hence", "Honestly", "Hopefully",
                                 "Immediately", "In addition", "In conclusion", "In fact", "In general", "In other words",
                                 "In particular", "In reality", "In short", "In the meantime", "In this case",
                                 "In this situation", "In truth", "Increasingly", "Indeed", "Interestingly", "Intimately",
                                 "Invariably", "Just", "Kindly", "Largely", "Later", "Likely", "Literally", "Logically",
                                 "Long", "Mainly", "Meanwhile", "Merely", "Most importantly", "Naturally", "Nearly",
                                 "Nevertheless", "Next", "Normally", "Not", "Notably", "Nowadays", "Obviously",
                                 "Occasionally", "Often", "On the contrary", "On the other hand", "Once", "Only",
                                 "Openly", "Originally", "Overall", "Particularly", "Perfectly", "Personally",
                                 "Possibly", "Practically", "Precisely", "Previously", "Probably", "Promptly", "Properly",
                                 "Quickly", "Quite", "Rarely", "Rather", "Readily", "Realistically", "Recently", "Regardless",
                                 "Regularly", "Relatively", "Remarkably", "Repeatedly", "Respectively", "Right", "Roughly",
                                 "Sadly", "Satisfactorily", "Second", "Seemingly", "Seriously", "Shortly", "Significantly",
                                 "Similarly", "Simultaneously", "Simply", "Slightly", "So", "Specifically", "Strictly",
                                 "Suddenly", "Supposedly", "Surely", "Temporarily", "Thankfully", "Thereafter", "Therefore",
                                 "Throughout", "Today", "Tomorrow", "Too", "Totally", "Truly", "Typically", "Ultimately",
                                 "Undoubtedly", "Unfortunately", "Usually", "Utterly", "Very", "Virtually", "Visibly", "Well",
                                 "Whatever", "Whenever", "Where", "Wherever", "Whether", "great", "good", "bad", "terrible",
                                 "excellent", "amazing", "wonderful", "horrible", "awful", "fantastic", "poor", "love",
                                 "hate", "like", "dislike", "experience", "place", "restaurant", "eat", "food", "dine",
                                 "order", "meal", "drink", "delicious", "tasty", "yummy", "taste", "guest", "patron",
                                 "customer", "client", "kitchen", "dish", "spoon", "fork", "knife", "glass", "beverage",
                                 "appetite", "hungry", "thirsty", "full", "empty", "bill", "check", "tip", "menu", "special",
                                 "recommend", "recommendation", "suggestion", "complaint", "critique", "review", "rating",
                                 "star", "visit", "try", "return", "come back", "visit", "guest", "visitor", "long",
                                 "address", "city", "town", "place", "restaurant", "establishment", "eatery", "bistro",
                                 "cafe", "diner", "joint", "pub", "grill", "steakhouse", "pizzeria", "bakery", "cafeteria",
                                 "deli", "food court", "gastropub", "tavern", "screaming", "crying", "babysitter",
                                 "accommodate", "toilet", "sink", "mirror", "soap", "distance", "mask", "vaccine", "covid",
                                 "pandemic", "coronavirus", "policy", "guideline", "requirement", "mandate", "regulation",
                                 "government", "protocol", "restriction", "limitation"]
            custom_stop_words_set = set(custom_stop_words)
            custom_stop_words = stop_words.union(custom_stop_words_set)
            custom_stop_words = list(custom_stop_words)

            lemmatizer = WordNetLemmatizer()

            def preprocess_text(text):
                words = word_tokenize(text.lower())
                words = [lemmatizer.lemmatize(word) for word in words if word.isalnum() and word not in stop_words]
                return ' '.join(words)

            df['preprocessed_text'] = df['Review'].apply(preprocess_text)

            # Sentiment analysis
            def analyze_sentiment(text):
                analysis = TextBlob(text)
                return analysis.sentiment.polarity
            df['sentiment'] = df['preprocessed_text'].apply(analyze_sentiment)

            # Convert numerical scores to sentiment labels
            threshold = 0.0
            df['sentiment_label'] = df['sentiment'].apply(
                lambda score: 'positive' if score > threshold else ('negative' if score < threshold else 'neutral'))

            # Split the data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(df['preprocessed_text'], df['sentiment_label'],
                                                                test_size=0.2, random_state=42)

            # Convert text data to TF-IDF features with custom stop words
            tfidf_vectorizer = TfidfVectorizer(stop_words=custom_stop_words)
            X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)

            # Train a Multinomial Naive Bayes model
            clf = MultinomialNB()
            clf.fit(X_train_tfidf, y_train)

            # Get the feature names (words) from the TF-IDF vectorizer
            feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

            # Get the log probabilities of each feature (word)
            log_probabilities = clf.feature_log_prob_

            # Check if there are both positive and negative classes
            if len(log_probabilities) > 1:
                # Calculate the difference between log probabilities for each word
                positive_class_log_probs = log_probabilities[1]  # Assumes 'positive' is class 1
                negative_class_log_probs = log_probabilities[0]  # Assumes 'negative' is class 0
                log_probability_diff = positive_class_log_probs - negative_class_log_probs
            else:
                # Handle the case where there is only one class (e.g., only neutral sentiment)
                log_probability_diff = log_probabilities[0]

            # Sort the log probability differences, feature names, and sentiments by importance
            sorted_indices = np.argsort(log_probability_diff)
            most_important_words = feature_names[sorted_indices]
            importance_scores = log_probability_diff[sorted_indices]
            word_sentiments = [
                "positive" if prob_diff > 0 else ("negative" if prob_diff < 0 else "neutral") for prob_diff in
                log_probability_diff]
            

            # Print the top 5 most important words for sentiment analysis for each restaurant
            output_string.append(f"<h3 style='padding-bottom: 1.5rem;padding-top: 1.5rem'>Top 5 most important words for {place_name}:</h3>")
            output_string.append(
                "<table style='display: inline-block; border-collapse: collapse; margin-right: 20px;' border='1'>")
            output_string.append(
                "<tr><th style='border: 1px solid #ddd; padding: 8px;'>Word</th><th style='border: 1px solid #ddd; padding: 8px;'>Importance Score</th></tr>")

            for word, importance in zip(most_important_words[-5:][::-1], importance_scores[-5:][::-1]):
                output_string.append("<tr>")
                output_string.append(f"<td style='border: 1px solid #ddd; padding: 8px;'>{word}</td>")
                output_string.append(f"<td style='border: 1px solid #ddd; padding: 8px;'>{importance:.2f}</td>")
                #output_string.append(f"<td style='border: 1px solid #ddd; padding: 8px;'>{sentiment}</td>")
                output_string.append("</tr>")

            output_string.append("</table>")

        # Join the list of strings to form the final output string
        formatted_output = '\n'.join(output_string)

        # Return the results
        return jsonify(result=formatted_output)

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        print(error_message)
        return jsonify(error=error_message), 500

if __name__ == "__main__":
    app.run(debug=True)
