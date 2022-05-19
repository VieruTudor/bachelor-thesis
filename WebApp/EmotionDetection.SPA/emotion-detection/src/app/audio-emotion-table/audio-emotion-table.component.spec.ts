import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AudioEmotionTableComponent } from './audio-emotion-table.component';

describe('AudioEmotionTableComponent', () => {
  let component: AudioEmotionTableComponent;
  let fixture: ComponentFixture<AudioEmotionTableComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [ AudioEmotionTableComponent ]
    })
    .compileComponents();
  });

  beforeEach(() => {
    fixture = TestBed.createComponent(AudioEmotionTableComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
