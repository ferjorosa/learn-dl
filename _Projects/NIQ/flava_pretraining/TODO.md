### Comentario sobre el pre-entrenamiento

Me acabo de acordar que para el preentrenamiento se samplea de diferentes datasets para hacer:
* Preentrenamiento de solo texto ("CCNews & BookCorpus" data) con probabilidad 0.15
* Preenentrenamiento de solo imagenes ("ImageNet-1k" data) con probabilidad 0.15
* Preentrenamiento multimodal ("PMD" data) con probabilidad 0.7

Creo que podemos tener el mismo framework de preentrenamiento y pasarle diferente dato ya sea de forma aleatoria o de forma estructurada, de esta manera hariamos los diferentes tipos de preentrenamientos. Por ejemplo, en el caso de brandbank, podemos contar con un dataset multimodal, pero en otros casos solo tenemos descripciones o imagenes, pues le pasamos esas imagenes para que el modelo vaya aprendiendo de forma unimodal. Asi podemos tambien mezclar diferentes dominios de negocio y diferentes tipos de imagenes/descripciones

#######################################################


El primer paso es  hacer una nueva version del builder que devuelva:
* La lista de imagenes asociadas al producto
* La lista de descripciones asociadas al producto
* Una lista de descripciones de tamaño X que no se corresponden con el producto en cuestion
  * Nota: Aqui me surgen dudas de como enfocarlo porque puede o ser siempre la misma descripcion "negativa" o una diferente para cada epoch

---------------

Hecho. La manera en la que lo he enfocado es devolver un numero de descripciones negativas de igual tamaño al numero de descripciones positivas. En cada iteracion vamos a samplear una descripcion de cada tipo.

Nota: Quizas para el caso donde solo tenemos una descipcion positiva, quizas debamos tener alguna mas negativa (un minimo) para que el modelo tenga mas variedad de dato.

--------------

#######################################################

La siguiente parte es entender como preparar el dato para su preentrenamiento.

Por lo que observo, si usamos el AutoProcessor de Flava solo nos prepara correctamente las imagenes para el MMM de imagenes.

* Para hacer el ITM, necesito que el input_ids este formado por 2 descripciones y tenga establecido el token_type_ids. Esto se puede hacer con un tokenizador, pasandole 2 descripciones.
* Para hcer el MLM, necesito utilizar el DataCollatoForLanguageModeling y (en mi opinion) fijarme que solo maskee los input_ids de la descripcion "verdadera" para evitar que se lie entre descripciones. Puede que no haga falta.
*

Para preparar el texto para MLM