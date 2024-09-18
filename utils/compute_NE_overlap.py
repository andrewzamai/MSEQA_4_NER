
label_sets = {
    'movie': ['actor', 'character', 'director', 'genre', 'plot', 'rating', 'ratings average', 'review', 'song', 'title', 'trailer', 'year'],
    'restaurant': ['amenity', 'cuisine', 'dish', 'hours', 'location', 'price', 'rating', 'restaurant name'],
    'ai': ['algorithm', 'conference', 'country', 'field', 'location', 'metrics', 'organisation', 'person', 'product', 'program lang', 'researcher', 'task', 'university'],
    'literature': ['award', 'book', 'country', 'event', 'literary genre', 'location', 'magazine', 'organisation', 'person', 'poem', 'writer'],
    'music': ['album', 'award', 'band', 'country', 'event', 'location', 'musical artist', 'musical instrument', 'music genre', 'organisation', 'person', 'song'],
    'politics': ['country', 'election', 'event', 'location', 'organisation', 'person', 'political party', 'politician'],
    'science': ['academic journal', 'astronomical object', 'award', 'chemical compound', 'chemical element', 'country', 'discipline', 'enzyme', 'event', 'location', 'organisation', 'person', 'protein', 'scientist', 'theory', 'university']
}

NEs_seen_by_GoLLIE = ['year',
                      'location', 'price', 'hours',
                      'product', 'country', 'person', 'organisation', 'location',
                      'event', 'person', 'location', 'organisation', 'country',
                      'event', 'country', 'location', 'organisation', 'person',
                      'person', 'organisation', 'location', 'election', 'event', 'country',
                      'person', 'organisation', 'country', 'location', 'chemical element', 'chemical compound', 'event'
]

if __name__ == '__main__':

    # Merge all the values into one list
    merged_list = [item for sublist in label_sets.values() for item in sublist]

    print([len(sublist) for sublist in label_sets.values()])

    # Convert the list to a set to remove duplicates
    unique_elements = set(merged_list)
    print(len(unique_elements))
    print(unique_elements)


    print("NEs seen by GOLLIE set")
    print(len(set(NEs_seen_by_GoLLIE)))
    print(set(NEs_seen_by_GoLLIE))

