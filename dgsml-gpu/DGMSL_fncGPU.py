from libraries import *
from utils import *
from Alexnetv2 import zero_grad, update
from DataloaderGPU import *
import copy
from Alexnetv2 import zero_grad, update, cloned_state_dict



def train_MAML(args, FT, CLS, meta_train_idx, meta_test_idx, ce_loss, optimizer1, optimizer2, batImageGenTrains, batImageGenTrains_metatest):
    beta0 = args.SSL_coef
    beta1 = args.gloabl_coef
    beta2 = args.class_wise_coef
    meta_step_size = args.meta_lr
    
    meta_train_loss = 0
    meta_test_loss = 0
    total_loss = 0
    centroids_tr = []
    dist_mat_tr = []
    
    global_loss = 0
    class_wise_loss = 0
    dist_mat_val = []    
                
    Loss_test = 0
    centroids_val = []

    for i in meta_train_idx:
        FT.train()
        CLS.train()
        domain_a_x, domain_a_y, domain_a_ux, domain_a_uy = batImageGenTrains[i].get_images_labels_batch(args)   
        x_subset_a = torch.from_numpy(domain_a_x.astype(np.float32))
        y_subset_a = torch.from_numpy(domain_a_y.astype(np.int64))
        x_subset_a = Variable(x_subset_a, requires_grad=False).cuda()
        y_subset_a = Variable(y_subset_a, requires_grad=False).long().cuda()

        x_subset_ua = torch.from_numpy(domain_a_ux.astype(np.float32))
        y_subset_ua = torch.from_numpy(domain_a_uy.astype(np.int64))
        x_subset_ua = Variable(x_subset_ua, requires_grad=False).cuda()
        y_subset_ua = Variable(y_subset_ua, requires_grad=False).long().cuda()

        feat_a = FT(x_subset_a)
        logits_a = CLS(feat_a)
        feat_ua = FT(x_subset_ua)
        logits_ua = CLS(feat_ua)
        l_task = ce_loss(logits_a, y_subset_a)

        unlab_class = torch.nn.functional.softmax(logits_ua, dim=1).argmax(1)
        ctr_l = class_centroid(args, feat_a, y_subset_a)

        
        score = scoring(torch.nn.functional.softmax(logits_ua, dim = 1))        
        m_feat_ua = score.view(-1,1).cuda()*feat_ua
        
        cat_F_lu, cat_Y_lu = cat(feat_a, y_subset_a, m_feat_ua, unlab_class)
       
        ctr_lu = class_centroid_lu(args, cat_F_lu, cat_Y_lu)     
    

        centroids_tr.append(ctr_lu)
 
        l_sl = torch.norm(ctr_l-ctr_lu) 
        
        train_loss = l_task + beta0*l_sl
        
        meta_train_loss = meta_train_loss + train_loss

    FT.zero_grad()
    grads_FT = torch.autograd.grad(meta_train_loss, FT.parameters(), create_graph=True)
    fast_weights_FT = cloned_state_dict(FT)
    
    adapted_params = OrderedDict()
    for (key, val), grad in zip(FT.named_parameters(), grads_FT):
        adapted_params[key] = val - meta_step_size * grad
        fast_weights_FT[key] = adapted_params[key]   

    CLS.zero_grad()
    grads_CLS = torch.autograd.grad(meta_train_loss, CLS.parameters(), create_graph=True)
    fast_weights_CLS = cloned_state_dict(CLS)

    adapted_params = OrderedDict()
    for (key, val), grad in zip(CLS.named_parameters(), grads_CLS):
        adapted_params[key] = val - meta_step_size * grad
        fast_weights_CLS[key] = adapted_params[key]   

    dist_mat_tr = row_pairwise_distances(centroids_tr)    

    for j in meta_test_idx:

        domain_b_x, domain_b_y, domain_b_ux, domain_b_uy = batImageGenTrains_metatest[j].get_images_labels_batch(args)
        x_subset_b = torch.from_numpy(domain_b_x.astype(np.float32))
        y_subset_b = torch.from_numpy(domain_b_y.astype(np.int64))
        x_subset_b = Variable(x_subset_b, requires_grad=False).cuda()
        y_subset_b = Variable(y_subset_b, requires_grad=False).long().cuda()

        x_subset_ub = torch.from_numpy(domain_b_ux.astype(np.float32))
        y_subset_ub = torch.from_numpy(domain_b_uy.astype(np.int64))
        x_subset_ub = Variable(x_subset_ub, requires_grad=False).cuda()
        y_subset_ub = Variable(y_subset_ub, requires_grad=False).long().cuda()
        

        feat_b = FT(x_subset_b, fast_weights_FT)
        feat_ub = FT(x_subset_ub, fast_weights_FT)

        logits_b = CLS(feat_b, fast_weights_CLS)
        logits_ub = CLS(feat_ub, fast_weights_CLS)

        unlab_class_b = torch.nn.functional.softmax(logits_ub, dim=1).argmax(1)
        ctr_l_b = class_centroid_lu(args, feat_b, y_subset_b)

        score_b = scoring(torch.nn.functional.softmax(logits_ub, dim=1))
        m_feat_ub = score_b.view(-1,1).cuda()*feat_ub
        cat_F_lu_b, cat_Y_lu_b = cat(feat_b, y_subset_b, m_feat_ub, unlab_class_b)
        ctr_lu_b = class_centroid_lu(args, cat_F_lu_b, cat_Y_lu_b)

        Loss_test = Loss_test + ce_loss(logits_b, y_subset_b)
        centroids_val.append(ctr_lu_b)

    dist_mat_val = row_pairwise_distances(centroids_val)
    global_loss = global_alignment(dist_mat_tr, dist_mat_val)
    meta_test_loss = Loss_test + beta1*global_loss

    total_loss = meta_train_loss + meta_test_loss

    optimizer1.zero_grad()
    optimizer2.zero_grad()
    total_loss.backward()
    optimizer1.step()
    optimizer2.step()        
        
    return total_loss


def test_fcn(args, batImageGenTest, FT, CLS):

    FT.eval()
    CLS.eval()

    if batImageGenTest is None:
        batImageGenTest = BatchImageGenerator(flags=flags, file_path='', stage='test', metatest=False, b_unfold_label=True)

    images_test = batImageGenTest.images
    labels_test = batImageGenTest.labels
    threshold = 100
    if len(images_test) > threshold:

        n_slices_test = math.ceil(len(images_test) / threshold)
        indices_test = []
        for per_slice in range(n_slices_test - 1):
            indices_test.append(math.ceil(len(images_test) * (per_slice + 1) / n_slices_test))
        test_image_splits = np.split(images_test, indices_or_sections=indices_test)

        test_image_splits_2_whole = np.concatenate(test_image_splits)
        assert np.all(images_test == test_image_splits_2_whole)

        test_image_preds = []
        for test_image_split in test_image_splits:
            test_image_split = get_image(args, test_image_split)
            images_test_split = torch.from_numpy(np.array(test_image_split, dtype=np.float32))
            images_test_split = Variable(images_test_split, requires_grad=False).cuda()

            Feat_ts = FT(images_test_split)
            classifier_out = CLS(Feat_ts)
            test_image_preds.append(classifier_out.cpu().detach().numpy())
        preds = np.concatenate(test_image_preds)
    else:
        images_test = get_image(args, images_test)
        images_test = torch.from_numpy(np.array(images_test, dtype=np.float32))
        images_test = Variable(images_test, requires_grad=False).cuda()
        Feat_ts = FT(images_test)
        classifier_out = CLS(Feat_ts)
        preds = classifier_out.cpu().detach().numpy()
    
    accuracy = compute_accuracy(predictions=preds, labels=labels_test)

    return accuracy

def validate_workflow(args, FT, CLS, batImageGenVals):

    accuracies = []
    for count, batImageGenVal in enumerate(batImageGenVals):
        accuracy_val = test_fcn(args, batImageGenVal, FT, CLS)

        accuracies.append(accuracy_val)

    mean_acc = np.mean(accuracies)

    return mean_acc


def heldout_test(args, FT, CLS, batImageGenTests):

    accuracies = []
    for count, batImageGenTest in enumerate(batImageGenTests):
        accuracy_test = test_fcn(args, batImageGenTest, FT, CLS)

        accuracies.append(accuracy_test)

    mean_acc = np.mean(accuracies)
        
    return mean_acc
