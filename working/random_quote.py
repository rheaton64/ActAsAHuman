import random

quotes = [
    ('Believe you can and you are halfway there.', 'Theodore Roosevelt'),
    ('Your time is limited, do not waste it living someone else life.', 'Steve Jobs'),
    ('You miss 100% of the shots you do not take.', 'Wayne Gretzky'),
    ('The best way to predict the future is to invent it.', 'Alan Kay')
]

def random_quote():
    quote, author = random.choice(quotes)
    return f'\"{quote}\" - {author}'

if __name__ == '__main__':
    print(random_quote())