    def training_step(self, batch: Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor, torch.Tensor, List[torch.Tensor]], _) -> torch.Tensor:
        '''
        Train
        Tabular-imaging contrastive learning
        Tabular reconstruction learning
        '''
        im_views, tab_views, y, _, original_tab = batch

        # =======================================  itc    =======================================================================
        # Augmented image and unagumented tabular
        z0, image_embeddings = self.forward_imaging(im_views[1]) 
        z1, tabular_embeddings = self.forward_tabular(tab_views[0])
        loss_itc, logits, labels = self.criterion_train_itc(z0, z1, y)
        self.log(f"multimodal.train.ITCloss", loss_itc, on_epoch=True, on_step=False)

        # =======================================  itm  =======================================================================
        loss_itm, logits_itm, labels_itm = self.cal_image_tabular_matching_loss(image_embeddings, tabular_embeddings, logits)
        self.log(f"multimodal.train.ITMloss", loss_itm, on_epoch=True, on_step=False)

        # =======================================  tr    =======================================================================
        # masked tabular 
        mask, mask_special = tab_views[2], tab_views[3]
        _, tabular_embeddings = self.forward_tabular(tab_views[1], mask=mask, mask_special=mask_special)
        z2, multimodal_embeddings = self.forward_multimodal(tabular_features=tabular_embeddings, image_features=image_embeddings)
        loss_tr, pred_cat, target_cat, mask_cat = self.criterion_tr(z2,original_tab,mask=mask)
        self.log(f"multimodal.train.TRloss", loss_tr, on_epoch=True, on_step=False)
        
        if len(im_views[0])==self.hparams.batch_size:
            self.calc_and_log_train_embedding_acc(logits=logits, labels=labels, modality='multimodal')
            self.calc_and_log_train_cat_embedding_acc(logits=pred_cat, labels=target_cat, mask=mask_cat, modality='multimodal')
            self.calc_and_log_train_itm_acc(logits=logits_itm, labels=labels_itm, modality='multimodal')
        
        loss = (loss_itc + loss_tr + loss_itm)/3.0
        # loss = (loss_itc + loss_tr)/3.0
        self.log(f"multimodal.train.loss", loss, on_epoch=True, on_step=False)
        return {'loss':loss, 'embeddings': multimodal_embeddings, 'labels': y, }