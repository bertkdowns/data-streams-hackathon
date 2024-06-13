from capymoa.evaluation.evaluation import *


def delayed_prequential_evaluation(
        stream, learner, max_instances=None, window_size=1000, optimise=True, store_predictions=False, store_y=False, delay=1
):
    """
    Calculates the metrics cumulatively (i.e. test-then-train) and in a window-fashion (i.e. windowed prequential evaluation).
    Returns both evaluators so that the caller has access to metric from both evaluators.
    """
    stream.restart()
    # if _is_fast_mode_compilable(stream, learner, optimise):
    #     return _prequential_evaluation_fast(stream, learner, max_instances, window_size,store_y=store_y, store_predictions=store_predictions)

    delayed_instances = []
    
    predictions = None
    if store_predictions:
        predictions = []

    ground_truth_y = None
    if store_y:
        ground_truth_y = []

    # Start measuring time
    start_wallclock_time, start_cpu_time = start_time_measuring()
    instancesProcessed = 1

    evaluator_cumulative = None
    evaluator_windowed = None
    if stream.get_schema().is_classification():
        evaluator_cumulative = ClassificationEvaluator(
            schema=stream.get_schema(), window_size=window_size
        )
        if window_size is not None:
            evaluator_windowed = ClassificationWindowedEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
    else:
        if not isinstance(learner, MOAPredictionIntervalLearner):
            evaluator_cumulative = RegressionEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
            if window_size is not None:
                evaluator_windowed = RegressionWindowedEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
        else:
            evaluator_cumulative = PredictionIntervalEvaluator(
                schema=stream.get_schema(), window_size=window_size
            )
            if window_size is not None:
                evaluator_windowed = PredictionIntervalWindowedEvaluator(
                    schema=stream.get_schema(), window_size=window_size
                )
    while stream.has_more_instances() and (
            max_instances is None or instancesProcessed <= max_instances
    ):
        instance = stream.next_instance()

        prediction = learner.predict(instance)

        if stream.get_schema().is_classification():
            y = instance.y_index
        else:
            y = instance.y_value

        evaluator_cumulative.update(y, prediction)
        if window_size is not None:
            evaluator_windowed.update(y, prediction)
        
        if len(delayed_instances) == delay:
            learner.train(delayed_instances.pop(0)) 
        
        delayed_instances.append(instance)

        # Storing predictions if store_predictions was set to True during initialisation
        if predictions is not None:
            predictions.append(prediction)

        # Storing ground-truth if store_y was set to True during initialisation
        if ground_truth_y is not None:
            ground_truth_y.append(y)

        instancesProcessed += 1

    # Stop measuring time
    elapsed_wallclock_time, elapsed_cpu_time = stop_time_measuring(
        start_wallclock_time, start_cpu_time
    )

    # Add the results corresponding to the remainder of the stream in case the number of processed
    # instances is not perfectly divisible by the window_size (if it was, then it is already be in
    # the result_windows variable). The evaluator_windowed will be None if the window_size is None.
    if (
            evaluator_windowed is not None
            and evaluator_windowed.get_instances_seen() % window_size != 0
    ):
        evaluator_windowed.result_windows.append(evaluator_windowed.metrics())

    results = {
        "learner": str(learner),
        "cumulative": evaluator_cumulative,
        "windowed": evaluator_windowed,
        "wallclock": elapsed_wallclock_time,
        "cpu_time": elapsed_cpu_time,
        "max_instances": max_instances,
        "stream": stream,
        "predictions": predictions,
        "ground_truth_y": ground_truth_y
    }

    return results