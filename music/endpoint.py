predictor = pytorch_estimator.deploy(
    initial_instance_count=1, instance_type='ml.m5.large')

endpoint_name = predictor.endpoint_name
print("Created endpoint: {}".format(endpoint_name))
