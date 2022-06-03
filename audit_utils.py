import time
from datetime import datetime

def save_results(results, output_path, start_time, args, info):
    end_time = time.time()

    output = open(output_path, "a")

    output.write("Audit Run Completed at: {}\n".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    output.write("Total time taken for this run: {} \n".format(time.strftime("%H:%M:%S", time.gmtime(end_time - start_time))))

    model_name, analysis_eps = info
    
    output.write("Audited Model: {}\nAnalysis Epsilon: {}\nNumber of Trials: {}\n".format(model_name, analysis_eps, args.num_trials))
    print("Audited Model: {}\nAnalysis Epsilon: {}\nNumber of Trials: {}\n".format(model_name, analysis_eps, args.num_trials))

    thresh, eps, acc = results 



    output.write("At threshold={}, epsilon={}\n".format(thresh, eps))
    output.write("The best accuracy at distinguishing poisoning is {}\n".format(acc))
    print("At threshold={}, epsilon={}\n".format(thresh, eps))
    print("The best accuracy at distinguishing poisoning is {}\n".format(acc))

    output.write("------------\n")
    output.close()