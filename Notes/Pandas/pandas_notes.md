### How to transform two Pandas columns into a dictionary

```python
df_dict = dict(zip(df.column_1, df.column_2))
```

### How to reverse the key/value of a dictionary

```python
inv_map = {v: k for k, v in my_map.items()}
```

### How to iterate over rows in Pandas

```python
for index, row in df.iterrows():
    print(row['column_1'], row['column_2'])
```

### How to avoid `SettingWithCopyWarning`

```python

# WRONG - Will throw a warning
df_ogrds[df_ogrds[subbrand_label].isin(df_char_normalization_map_subbrand["OLD"])][subbrand_label] = "OUTRAS MARCAS"

# CORRECT
df_ogrds.loc[df_ogrds[subbrand_label].isin(df_char_normalization_map_subbrand["OLD"]), subbrand_label] = "OUTRAS MARCAS"

```

There is other example that I have seen where my only solution has been to use copy()

```python

df_ogrds = df_ogrds_old[[description_label, category_label, brand_label]]

# WRONG - Will throw a warning
df_ogrds[description_label] = df_ogrds[description_label].apply(lambda x: normalize(str(x)))


# STILL WRONG (I belive due to the previous column slice)
df_ogrds.loc[:, description_label] = df_ogrds[description_label].apply(lambda x: normalize(str(x)))

# CORRECT (with copy())
df_ogrds_2 = df_ogrds_old[[description_label, category_label, brand_label]].copy()
df_ogrds_2[description_label] = df_ogrds_2[description_label].apply(lambda x: normalize(str(x)))

```

### How to count missing values per column

````python
missing_data = df.isnull()

true_counts = [(column, missing_data[column].values.sum()) for column in missing_data.columns]
false_counts = [(column, (~missing_data[column].values).sum()) for column in missing_data.columns]

true_counts.sort(key=lambda x:x[1], reverse = True)
```

### How to replace pandas.append()?

```python

confidence_list = []
brand_predicted_list = []
description_list = []
brand_list = []
for brand, description in df_val_list_tuple:
    prediction= model.predict(description)
    brand_predicted = prediction[0][0]
    confidence = round(prediction[1][0], 2)

    brand_predicted_list.append(brand_predicted)
    confidence_list.append(confidence)
    description_list.append(description)
    brand_list.append(brand)

df_val_results_dict = {
        "DESCRIPTION": description_list,
        "BRAND": brand_list,
        "BRAND_PREDICTED": brand_predicted_list,
        "CONFIDENCE": confidence_list
    }
df_val_results = pd.DataFrame.from_dict(df_val_results_dict)

print(df_val_results.shape)
df_val_results.head(2)

```

### Is one column the substring of another column?

Example with brand column and subbrand column. The idea is to find if the subbrand column contains elements with brand in its name. 

**Careful:** It is not perfect because there could be parts of the brand name in the subbrand but not the whole brand name and this would not get it...

```python
df[df.apply(lambda x: x[brand_label] in x[subbrand_label], axis=1)]
```