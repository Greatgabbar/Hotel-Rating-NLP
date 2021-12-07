def modalFunction():
    yelp = pd.read_csv('yelp.csv')
    yelp.head()
    yelp['text length'] = yelp['text'].apply(len)
    sns.set_style('white')
    g = sns.FacetGrid(yelp,col='stars')
    g.map(plt.hist,'text length')
    sns.boxplot(x='stars',y='text length',data=yelp,palette='rainbow')
    sns.countplot(x='stars',data=yelp,palette='rainbow')
    stars = yelp.groupby('stars').mean()
    stars.corr()
    yelp_class = yelp
    X = yelp_class['text']
    y = yelp_class['stars']
    cv = CountVectorizer()
    X = cv.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.3,random_state=101)
    nb = MultinomialNB()
    nb.fit(X_train,y_train)
    predictions = nb.predict(X_test)
    print(confusion_matrix(y_test,predictions))
    print('\n')
    print(classification_report(y_test,predictions))
    inp=["average"]
    b=cv.transform(inp)
    out=nb.predict(b)
    print(out)